# 02 · 四能力轴逐项对账

## 1. 读表规则

本表不使用模糊的“支持/不支持”，而用三种机制强度：

- **D（direct）**：该工作在自己的实验设置内直接实现并测量这条能力；
- **P（partial）**：有相邻机制，但缺 Volvence 定义中的关键条件；
- **—**：不处理或与该轴相冲突；
- **E（evaluation）**：只提供测量、诊断或反例，不实现能力。

`D` 也不表示 production pass。例如 ReFT 对 Steerable 是 D，但没有 Volvence 的 PE-only gate、
snapshot、norm cap、WiringLevel 和 rollback。

## 2. 总矩阵

| 工作 | 年份 / 状态 | Appendable | Readable | Learnable | Steerable | 最高机制级 | 控制 locus |
|---|---|---:|---:|---:|---:|---:|---|
| Function Vectors | 2024 ICLR | — | D | — | D | M2 | 外部提取/注入，内部函数表示 |
| Refusal Direction | 2024 NeurIPS | — | D | — | D | M2 | 外部单方向增删 |
| ReFT / LoReFT | 2024 NeurIPS | — | P | P | D | M2 | 离线学习、推理时表征干预 |
| CAST | 2025 ICLR | — | P | — | D | M3 | 静态 condition reader + gate |
| Steering off Course | 2025 ACL | — | E | — | E | M2 反证 | 跨模型可靠性审计 |
| Understanding (Un)Reliability | 2025 preprint/workshop | — | E | — | E | M2 反证 | 几何可控性诊断 |
| Persona Vectors | 2025 arXiv | — | D | P | D | M2 | trait 监控、预防和反向 steering |
| FASB | 2025 NeurIPS | — | P | — | D | M3 | 生成中动态必要性/强度/回退 |
| PID Steering | 2025 arXiv | — | P | — | D | M3 | 跨层反馈误差控制 |
| ETA Internal RL | 2025 arXiv | — | P | D | D | M4（仿真） | latent controller + learned termination |
| Titans | 2024/25 arXiv | P | — | P | — | M3 | 测试时 neural-memory update |
| ATLAS | 2025 arXiv | P | — | P | — | M3 | 历史感知的高容量 neural memory |
| Nested Learning / HOPE | 2025 NeurIPS | P | P | P | — | M3 | 多频嵌套优化与自修改 memory |
| SEAL | 2025 NeurIPS | P | — | D | — | M4（离线/慢） | 自生成 edit + 权重更新 |
| TTT-E2E | 2025 arXiv | P | — | P | — | M3 | 序列内 NTP 更新快权重 |
| Spurious Forgetting | 2025 ICLR | E | E | E | — | M2 诊断 | 对齐失活 vs 知识丢失 |
| J-space | 2026 research thread | — | D | — | D | M2 / IC-2 | 模型内生调制 + 外部 lens/swap |
| Emotion Concepts | 2026 research thread | — | D | — | D | M2 | 局部 operative concept |
| Natural Language Autoencoders | 2026 research thread | — | P | P | P | M2 | 文本瓶颈重建 activation |
| Causality ≠ Invariance | 2026 ICLR | — | D/E | — | D/E | M2 | 跨格式 Concept Vector |
| FaithSteer-BENCH | 2026 arXiv | — | E | — | E | M3 反证 | 部署三门 stress test |
| TACT | 2026 arXiv | — | D | — | D | M3 | agent step reader + selective correction |
| Activation-LQR | 2026 arXiv | — | P | — | D | M3 | activation setpoint feedback |
| Per-instance Multi-layer | 2026 arXiv | — | P | — | D | M3 | 逐输入选层、方向与 adaptive-K |
| Forecasting Side Effects | 2026 arXiv | — | E | — | E | M3 审计 | 干预前 cross-effect 预测 |
| CL-Bench | 2026 arXiv | E | — | E | — | M3 评测 | 真实 stateful gain metric |
| TACT + memory（公开组合） | 未发现 | — | — | — | — | — | 尚无证据 |
| 严格四轴闭环（公开） | 未发现 | — | — | — | — | — | 尚无证据 |

## 3. Appendable：行业把“写进去”做到了，但没有把“该不该写”解决

### 3.1 已成立的局部机制

- Titans / ATLAS：运行中更新 neural memory；
- Nested Learning / HOPE：按不同频率组织可更新 memory/optimizer；
- TTT-E2E：把当前 context 通过 NTP 压进快权重；
- SEAL：生成 self-edit 并形成持续权重更新；
- agent memory 系统：把经验写入外部数据库、图或摘要。

### 3.2 为什么这里只标 P

Volvence 的 Appendable 不只是“状态发生变化”，还要求：

1. 跨 session 恢复；
2. 分层时间尺度；
3. 唯一 owner；
4. 冲突、陈旧、删除和 consent；
5. write gate 与 rollback；
6. 后续行为证据，而非只降低 language-model loss。

现有 neural memory 多在一条长序列内评估；weight-edit 路线又把删除/归因/回滚变成难题。CL-Bench 的
负结果进一步说明，Appendable 的瓶颈不是容量，而是**选择性写入、正确抽象和及时失效**。

### 3.3 Volvence 的真实差距

当前已具 CMS、hydration、checkpoint 与多层机制，但仍缺：

- memory-only 相对 naive ICL / long-context 的稳定增益；
- ModificationGate 对所有沉淀写面的一致覆盖；
- stale-belief、conflict、delete 后的行为级复测；
- relationship state 经过跨 session 恢复后对真实选择的因果贡献。

## 4. Readable：从“线性可读”进展到“可言说且因果负载”，但不变性仍失败

### 4.1 已成立

- task、refusal、persona、emotion、agent drift 和 unspoken intermediate 都能从 residual 中读出；
- J-space 把 readout、swap、ablation 和模型自发调制放在同一组实验里；
- NLA 开始摆脱预定义标签，用自然语言瓶颈提出 activation 解释；
- Concept Vectors 表明可以显式优化跨格式稳定性。

### 4.2 仍未成立

- 一个 readout 是否覆盖所有目标状态；
- 复杂关系结构是否能由 token-indexed concept bag 表示；
- model upgrade 后 artifact 是否仍有效；
- 可读状态是否可校准为策略输入；
- local operative emotion 是否可区分于 persistent relationship state；
- consumer 是否能只读 publisher snapshot，而不重建隐状态。

### 4.3 对 Volvence 的裁决

外部证据强化了 Readable 的科学可行性，却同时抬高了 formal 门槛：仅有 heldout accuracy 已不够。
至少还需：

```text
同域判别力
+ 跨视图稳定性
+ causal patch / downstream sensitivity
+ calibration
+ model/layer/domain/version lineage
+ owner-published immutable snapshot
```

## 5. Learnable：这是外部拼图中最薄的一轴

### 5.1 看似 Learnable、但信号不同

| 工作 | 实际学习信号 | 与 PE-only 的差距 |
|---|---|---|
| Titans / ATLAS | token/memory mismatch、gradient surprise | memory-local，不是行动后真实 outcome |
| TTT-E2E | next-token prediction loss | 对当前上下文有效，不分 world/self 或行动归因 |
| SEAL | 更新后 downstream performance reward | 外部 task reward；慢；写权重 |
| ETA | environment sparse reward | Internal RL locus 相近，信号与对话域不同 |
| Persona preventative steering | 训练数据在 trait direction 上的 projection | 预测 drift，不是在线 outcome credit |
| TACT | 离线 step label | gate 本身不持续更新 |

### 5.2 真正缺失的证据

公开材料中尚未看到如下完整链：

```text
同一真实 decision 的 steer/noop counterfactual
→ arm-independent N+1 target
→ Prediction Error owner
→ sparse credit
→ 只更新 gate policy
→ 后续 episode 改善
→ 跨 session checkpoint 恢复
```

Volvence 的 C1 契约与 S3-E 代理证据在结构上走得更远，但对话域 C3 formal 和用户可见行为层仍未过门。

## 6. Steerable：机制最强，可靠性债务也最大

### 6.1 行业已经完成的升级

```text
固定单向量
→ 学习式低秩 executor
→ condition gate
→ generation-state gate
→ per-instance layer / direction / dose
→ PID / LQR feedback
→ side-effect forecasting
```

这说明“冻结基底 + 内部控制”已从实验技巧发展为一个工程路线。

### 6.2 关键断层

1. **样本异质性**：平均正效应掩盖 individual reverse effect；
2. **表示异质性**：同一概念跨格式可能近正交；
3. **层异质性**：最佳层随输入变化；
4. **剂量异质性**：过强或过多层会伤害已有正确输入、破坏流畅性；
5. **副作用耦合**：目标行为变化会非对称影响其他行为；
6. **后端差异**：transformers hook 的成立不保证 vLLM/SGLang production path 等价；
7. **信号错位**：追踪 semantic setpoint 不等于改善真实 N+1 outcome。

### 6.3 Volvence 当前位置

仓库已有严格的 no-free-bias、norm cap、strict noop、artifact lineage、SHADOW isolation 和有序
promotion 设计，这是公开方法常缺的治理层。当前缺的不是更多契约，而是：

- substrate 能否应用已披露/已读出策略；
- 对话域 reader/executor/gate 的真实 formal；
- 用户可见行为 effect；
- per-instance layer/dose 是否值得扩大 action space；
- vLLM 可验证残差出口；
- capability-tax 与 side-effect audit。

## 7. 哪些组合最接近我们

| 组合 | 覆盖 | 仍断在哪里 |
|---|---|---|
| J-space + CAST | 内生可读表征 + 条件写入 | gate 不学习、无持久状态 |
| J-space + TACT | 内部状态 + 长轨迹真实任务 | 状态标签离线、无跨 episode 学习 |
| ReFT + per-instance gate | 学习式 executor + 逐输入控制 | gate 不是 PE-credit，memory 缺失 |
| Titans/HOPE + J-space | 多频写入 + 命名工作空间 | 两者未在同一系统中因果闭合 |
| ETA + Titans | latent policy learning + neural memory | 公开证据中未组合；reward/治理不符 |
| TACT + CL-Bench | 真实 agent outcome + stateful gain metric | 尚未有人公开完成组合实验 |
| Volvence 当前 | A 机制 + R/S/L 代理 + SHADOW owner 链 | 真实 substrate/readout floor、C3/B3、behavioral N+1、ACTIVE |

## 8. 对外定位建议

不再使用：

> “Volvence 首次发现 LLM 内部可以控制行为。”

建议使用：

> “公开研究已证明 LLM 内部存在可读、可写且具有因果作用的控制表征。Volvence 研究的是尚未闭合的
> 系统问题：如何让这些控制表征在跨 session 状态上，由 Prediction Error 的可审计信用学会逐实例择时，
> 并以有界、可撤销的方式改变真实下一拍。”

只有 [`05_EVIDENCE_ROADMAP.md`](05_EVIDENCE_ROADMAP.md) 中四臂纵向主实验和 promotion gate 通过后，
才能把“研究的是”升级为“已经证明”。
