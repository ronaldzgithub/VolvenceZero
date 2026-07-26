# 可塑性丧失：范围收窄、失效机制与分级解法

> 承接 [`02_VZ_DELTA.md`](02_VZ_DELTA.md) §3.A。
> 本文所有代码判断在 2026-07-26 `main`（`d56bdf2`）上逐文件核对。
> Status: **设计提案**，不是已批准的 spec，不进主链。

---

## 0. 先纠正上一轮的范围判断

[`02_VZ_DELTA.md`](02_VZ_DELTA.md) §3.A 写的是"我们的学习型控制器、head、CMS 频段"全部继承这个缺陷。**这个范围说宽了。** 逐个核对实现后：

| 组件 | 实际形态 | 适用 Nature 可塑性丧失？ |
|---|---|---|
| **`CMSBandMLP` × 3**（online-fast / session-medium / background-slow） | **2 层残差 MLP**，`y = clamp(x + W1·tanh(W2·x))`，`d_hidden = max(latent_dim*2, 8)`，逐轮更新 + 动量 + replay，**跨会话持久** | **✅ 唯一真正在范围内的** |
| `BackendCMSBand`（`torch_cms_band.py`） | 同一套数学的后端加速版本 | ✅ 同一对象 |
| `TorchCausalZPolicy` / internal RL | 源码注释即 `"PPO trainer for the causal z-policy (real autograd, **offline**)"`；`__init__` 里 `self.policy = TorchCausalZPolicy(...)` **每次全新构造** | ❌ 离线 episodic，不跨运行累积 |
| `RewardingStateHeadState` | `weights: tuple[float,...]` + `bias` —— **线性** | ❌ 无隐藏单元 |
| `RegimeScoreLearner` | 注释即 `"Bounded **linear** head"` | ❌ |
| `DualTrackGateLearner` | 注释即 `"bounded online-SGD **linear**"` | ❌ |
| `ConsolidationScoreLearner` | 注释即 `"Bounded **linear** residual head"` | ❌ |
| `PeWriteGate` | 标量阈值 + ±0.10 envelope 夹 | ❌ |

**真实敞口是 3 个 CMS band MLP，不是"所有学习型 owner"。**

线性头没有隐藏单元可死、没有表示秩可塌——Nature 那篇讲的现象在它们身上不成立。它们有**另一种**退化（见 §5），但那是不同的失效，不该并进同一个指标。

这把问题从"架构性隐患"收窄成"一个具体模块的具体缺陷"。**并且它是 live 的**：`memory/store.py:107` 构造生产核心时写死 `mode="mlp"`。

---

## 1. 针对我们这套数值的精确失效机制

前向（`cms_band_mlp.py`）：

```
h = W2 @ x          x ∈ [0,1]^d_in      # _clamp 是 [0,1]，不是 [-1,1]
a = tanh(h)
y = clamp(x + W1 @ a)
```

反向的关键一行（`cms_band_mlp.py:211`）：

```python
grad_h.append(grad_a[j] * (1.0 - a[j] * a[j]))
```

**失效链条**：

1. 代码里**没有任何 weight decay 或权重范数约束**——W2 只被梯度驱动，可无界增长。
2. `x ∈ [0,1]` **恒为非负**，所以 `h_j = W2_j · x` 的符号基本由 W2_j 决定且不易反号 → 长跑下 `|h_j|` 单调偏大。
3. `|a_j| → 1` 时 `(1 - a_j²) → 0` → **该隐藏单元的 W2 行梯度归零 → 永久冻结**。
4. 冻结后 `a_j ≈ 常数`，它对输出的贡献 `W1[:,j] · a_j` 退化成一个**偏置**，不再是特征。
5. 逐个单元重复 → 有效隐藏单元数下降 → 正是论文说的"单元多样性丧失 + 有效秩下降"。

### 一个会让仪表永远显示健康的陷阱

我们的失效不是 **dead ReLU（a→0）**，是 **saturated tanh（|a|→1）**。

`Self-Normalized Resets`（2410.20098）的判据是"神经元发放率有效地降到零"——**ReLU / firing-rate 口径**。直接照搬会**系统性漏判我们的全部失效单元**。

**判据必须是 `|a_j| → 1` 且 `var(a_j) → 0`，不是 `a_j → 0`。**

---

## 2. 好消息：三个关键原语我们已经有了

1. **回滚**：`export_params()` / `restore_params()` 已存在，6 个参数组（state / state_mom / w2 / w1 / w2_mom / w1_mom）。R15 需要的东西不用新建。
2. **容量记账**：`parameter_count()` 已存在。
3. **输出保持的重初始化是免费的**：`self._w1 = [0.0] * (d_in * d_hidden)`，注释写明 *"W1 initialized to zero so the MLP starts as an identity map"*。

第 3 条很关键：**重初始化一个单元时，重置它的 W2 行、把 W1 列置零，网络计算的函数完全不变**——重初始化是**输出保持（output-preserving）**的，不会造成行为跳变。

Nature 的 continual backprop 正是这么做的（重置入边、出边置零），而**我们的残差 + W1 零初始化架构天然就是这个形状**。这让修复的风险远低于一般情况。

---

## 3. 分四级解法

### L0 — 仪表（必须先做，readout-only）

没有仪表就不该动任何权重。两层，匹配我们的时间尺度：

**Tier 1（每轮，O(d_hidden)，不需要线性代数）**

`a` 在 `forward` 与 `_apply_single_target_update` 里都已经算出来了，顺手累计即可：

- `saturation_fraction = #{j : |a_j| > 0.99} / d_hidden`（滚动窗口）
- 每单元激活方差 `var(a_j)`（Welford 在线方差，O(1)/单元）
- `effective_active_units = #{j : var(a_j) > ε}`

**Tier 2（background-slow 频段，真有效秩）**

`d_hidden ≤ 32`，Gram 矩阵只有 32×32：

```
G = AᵀA / n                    # A = 最近 n 轮的 a 向量堆叠
λ = eig(G)                     # 对称 Jacobi 特征值
p_k = λ_k / Σλ
erank = exp(−Σ p_k · log p_k)  # Nature 原式
```

`vz-memory` 是手写 `_matvec` 的零依赖 wheel，所以 Jacobi 也得手写——但 32×32 的对称特征值分解可控，且只在 background-slow 跑。

发布到 `MemorySnapshot.lifecycle_metrics`（`PeWriteGate` 的 readout 已经走这条路）。

**告警判据**：`erank` 或 `effective_active_units` 在 ≥N 轮窗口上**单调下降**。这也是一个干净的 kill condition 候选。

### L1 — 预防（最便宜的真修复）：约束 W2 行范数

病根是 W2 无界增长导致饱和。加一个**投影**（不是惩罚项），每次更新后：

```python
for j in range(d_hidden):
    n = norm(W2[j, :])
    if n > W2_ROW_MAX:
        W2[j, :] *= W2_ROW_MAX / n
```

- 与代码库既有惯用法一致（到处是 `_clamp`）
- 确定性、不引入随机性、天然可回滚（就是个夹子）
- **防止**进入饱和终态，而非事后修复
- 只新增一个常数 `W2_ROW_MAX`，且可以用 L0 的仪表来标定

**先做 L1 再考虑 L2。** 预防比修复便宜、风险低，且不需要 gate。

### L2 — 纠正（gated 的选择性重初始化）

当某单元连续 k 个结算窗口低效用（`var(a_j) < ε` 或持续饱和）时：

```python
W2[j, :]      ← 新的小随机值    # 入边：注入多样性
W1[:, j]      ← 0               # 出边：置零 → 输出保持
W2_mom[j, :]  ← 0
W1_mom[:, j]  ← 0               # 动量必须一起清，否则旧动量把它推回饱和
```

**这在结构上就是一次 R10 操作**，我们的 gate 词汇正好对得上（`credit-and-self-modification.md`）：

- `capacity_cost` = 本次重初始化的单元数
- `rollback_evidence` = 重初始化前的 `export_params()`
- `validation_delta` = 重初始化后 N 轮的 band 预测误差改善

**速率要比论文保守得多**：**每个结算窗口至多 1 个单元**，且必须连续 k 个窗口失败才触发。`d_hidden` 只有 8–32，动一个已经是 3%–12% 的容量。

#### ⚠️ 一个会让修复静默失效的实现陷阱

`_init_weight` 是**按索引的确定性哈希，不是 RNG**（`cms_math.py:36`）：

```python
scale * (((i * 2654435761 + 17) % 65537) / 32768.5 - 1.0)
```

如果重初始化时对同一个单元索引 `j` 再调一次 `_init_weight`，会拿到**与上次逐位相同的值** → **没有注入任何多样性 → 修复变成 no-op**，而仪表还会显示"我们已经重置过了"。

必须让一个**递增的重置计数器**参与哈希（例如按 `j + reset_counter * d_hidden` 取偏移），并把该计数器纳入 `export_params` / `restore_params`，以同时保住**确定性**与**回滚**。

这一条如果漏掉，整个 L2 会以最难发现的方式失败。

### L3 — 换激活（最后手段）

`Activation Function Design Sustains Plasticity`（ICLR 2026）说激活选择是"首要的、架构无关的杠杆"，给出 Smooth-Leaky / Randomized Smooth-Leaky 两个 drop-in。

**但对我们代价最高**：`update_with_replay` 的 docstring 明说它与 legacy `update` **逐位相等（bit-equal）**，且 *"the SHADOW vs ACTIVE protocol relies on"* 这一点。换激活会破坏该等价 → **既有的 SHADOW→ACTIVE 证据全部作废、需要重跑**。

**只有当 L0 的仪表证明 L1 + L2 不够时才动。**

---

## 4. 落地顺序

| 级 | 动作 | 风险 | 前置 |
|---|---|---|---|
| **L0** | `plasticity_readout`：saturation / 有效单元数 / erank，readout-only | 无 | — |
| **L1** | W2 行范数投影 | 低 | L0 有基线 |
| **L2** | gated 单元重初始化（**含重置计数器**） | 中 | L0 + L1 观察 ≥1 wave |
| **L3** | 换激活 | 高（作废 SHADOW 证据） | L0 证明 L1+L2 不足 |

按 `credit-and-self-modification.md` 的三阶段协议：L0 是 `readout-only`；L2 的触发判据走 `readout-with-acceptance → acceptance gate`，并配一个 rollback drill 测试（照 `tests/contracts/test_learned_baseline_rollback_drill.py` 的形状）。

**L0 + L1 合起来是低风险、无需 gate 的一次改动，且能覆盖大部分实际收益**（预防饱和 > 修复饱和）。L2 才需要动 gate 机制。

---

## 5. 线性头的退化是另一个问题（别混）

`RewardingStateHead` / `RegimeScoreLearner` / `DualTrackGateLearner` / `ConsolidationScoreLearner` / `PeWriteGate` 没有隐藏单元，上面的东西一概不适用。它们的退化模式是：

1. **权重顶到 clamp/envelope 边界后失去响应**（`PeWriteGate` 的 ±0.10 envelope 是显式的；其他头的 clamp 需各自核对）
2. **上游特征塌缩**——如果输入特征来自已退化的 CMS band，头自身健康也没用

对应仪表便宜得多：`fraction of weights at clamp boundary` + `输入特征方差`。

建议与 L0 同期做，但**必须作为独立指标**，不要合并成一个"可塑性"数字——它们是不同的失效，混在一起会让两边都读不出来。

---

## 6. 与前两轮归因表的合并

至此我们有了完整的三分归因（[`02_VZ_DELTA.md`](02_VZ_DELTA.md) §6 的 O 条）：

| 现象 | 归因 | 判定方法 | 来源 |
|---|---|---|---|
| 指标下降 | **对齐被掀翻**（知识还在） | 少量旧数据 replay 后恢复 / 底层冻结对照臂不降 | Spurious Forgetting（第一轮） |
| 指标下降 | **知识真丢了** | replay 后不恢复 | 经典灾难性遗忘 |
| 指标下降 | **网络学不动了** | `erank` / `effective_active_units` 持续下降；对新目标的拟合速度变慢 | Loss of Plasticity（本轮） |

**第三行没有 L0 的仪表就无法判定**——这是 L0 优先级最高的真正理由：它不只是修一个模块，它是让归因表可用的前提。
