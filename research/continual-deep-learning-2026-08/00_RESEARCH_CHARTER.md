# 00 · 研究章程与证据口径

## 1. 研究目标

本项目回答三个问题：

1. **业界是否独立发现了 LLM 内部控制面，证据强到哪一步？**
2. **与 Volvence 四能力轴相比，真正还没有解决的是什么？**
3. **哪些外部机制、指标、代码和负结果可以直接缩短我们的证据路线？**

研究对象不是宽泛的“continual learning”，而是以下交集：

```text
内部表征可读 / 可写
        ×
测试时或在线适应
        ×
条件化 / 反馈 / 策略控制
        ×
长期记忆与信用结算
```

## 2. 五种“内部控制”必须区分

“LLM 内部就可以实现控制”至少有五种强度不同的含义。若不区分，会把同一个词用于完全不同的证据。

| 级别 | 定义 | 代表证据 | 还不能推出 |
|---|---|---|---|
| IC-0 相关可读 | hidden state 能预测标签或行为 | 线性 probe、persona projection | 该方向有因果作用 |
| IC-1 外生因果写入 | 外部系统改 activation 后行为改变 | CAA、Function Vector、Refusal direction、ReFT | 模型会自己决定何时写 |
| IC-2 内生工作空间 | 模型因任务要求主动调制内部表征，并用它推理 | J-space directed modulation / reasoning | 跨 turn 持久、会从 outcome 学 |
| IC-3 条件反馈控制 | reader 根据当前内部状态决定是否、何处、写多大 | CAST、FASB、TACT、per-instance steering、PID/LQR | gate 的目标来自合法在线信用 |
| IC-4 持续学习闭环 | 真实结果结算后更新 gate/记忆，下一 episode 恢复并改进 | 尚无完整公开同类 | — |

Volvence 的目标是 IC-4；外界已较强地证明 IC-1/IC-2，并开始工程化 IC-3。

## 3. 四能力轴操作定义

本包服从仓库正式定义，不采用论文自己的宽松命名：

| 轴 | 本项目认定为直接证据的最低条件 | 常见伪替代 |
|---|---|---|
| Appendable | 新经历写入分层状态，能在后续 episode/session 恢复；写入、冲突和删除路径可说明 | 把当前 prompt 变长；单次 KV；不可恢复的临时梯度 |
| Readable | 从内部表示读出命名、校准、可版本化状态，并有独立效度或因果检查 | 文本关键词；模型自述；只报 probe accuracy |
| Learnable | 经历后的一级误差信号及下游信用实际改变后续策略/状态 | judge 分数直接当 reward；离线标签训练一个静态 probe |
| Steerable | 冻结基底上条件化、有界、strict-noop 的因果干预，最好能影响真实下一拍 | prompt 改写；always-on 大向量；在线改全模型权重 |

外部工作通常不实现 Volvence 的 owner/snapshot/WiringLevel。因此 [`02_FOUR_AXIS_CROSSWALK.md`](02_FOUR_AXIS_CROSSWALK.md)
采用“直接机制 / 邻接机制 / 未覆盖”而非笼统的 pass/fail。

## 4. 证据等级

### 4.1 机制证据等级

| 等级 | 含义 | 典型证据 |
|---|---|---|
| M0 | 概念或架构提案 | 无受控实验 |
| M1 | 相关读出 | probe、projection、聚类；无干预 |
| M2 | 因果局部机制 | swap、ablation、patch、matched intervention |
| M3 | 条件/逐实例控制 | gate、动态强度、真实轨迹或多模型 heldout |
| M4 | 在线学习闭环 | outcome→credit→policy update→后续改善 |
| M5 | 长期部署闭环 | 跨 session、真实用户、SLO、安全、删除与回滚 |

### 4.2 发表与复现等级

| 标记 | 含义 |
|---|---|
| V1 | 同行评审主会论文 |
| V2 | arXiv / 实验室 research thread，方法与结果公开 |
| V3 | 公司工程报告或官方代码，自报结果 |
| V4 | 二手解读，仅用于发现线索，不承载结论 |

本项目的实质结论只由 V1–V3 一手材料承载；V3 的吞吐结果不能升级为 M3/M4 行为证据。

## 5. 因果主张最低审计

任何“这个内部方向控制了行为”的主张至少检查：

1. 是否有 matched noop / random direction / norm-matched 对照；
2. 是否把 probe 权重误当作 intervention direction；
3. 是否跨 seed、跨 layer 或跨模型；
4. 是否报告 already-correct 样本被破坏的数量；
5. 是否测 unrelated capability tax；
6. 是否存在 prompt format、language、role prompt 或 paraphrase 变化；
7. 是否在终局结果上有效，而非只移动内部投影；
8. 是否区分目标行为改变与输出 token 偏置；
9. 是否说明干预剂量、norm、位置和时机；
10. 是否能 strict noop 并可恢复原行为。

## 6. “可以被我们所用”的判定标准

候选机制按五类裁决：

- **Adopt**：可直接采用其指标、对照或审计方法，不改变系统 owner。
- **Adapt**：核心机制有价值，但必须改成冻结基底、PE-only、bounded、snapshot/lineage 形式。
- **Baseline**：不能进入主链，但必须成为强对照，否则 Volvence 的增益没有归因力。
- **Watch**：理论方向重要，当前证据或工程成熟度不足。
- **Reject**：与 R2/R8/R10/R12/R15 冲突，或其主张被关键负结果击穿。

每个 Adopt/Adapt 候选还必须回答：

```text
唯一 owner 是谁？
输入输出快照是什么？
学习信号来自哪里？
失败时怎样单字段回滚？
什么实验会证伪其价值？
```

## 7. 检索范围与截止线

- 时间重点：2024-01 至 2026-08-22；早期工作仅在构成直接谱系时纳入。
- 来源优先级：会议论文 / 官方论文页 → arXiv 原文 → 官方代码 → 官方工程报告。
- 技术主题：representation reading、activation steering、conditional control、feedback control、
  test-time memory、continual learning、internal RL、inference infrastructure。
- 不把融资稿、产品营销、二手媒体和无实验观点作为核心证据。
- 对“目前没有公开系统”一类结论采用开放世界口径：意为本轮可检索的一手公开材料中未发现，
  不声称私有系统必然不存在。

## 8. 与 Volvence 当前状态的冻结基线

本项目按 2026-08-22 仓库台账对账：

- 代理环境已有“读得到 + 扳得动 + 学会何时扳”的 S3-E 机制证据；
- runtime steering 三 owner 已 SHADOW 接线，production ACTIVE 未授权；
- 对话域 Readable、C3/B3 formal、用户可见 residual steering、真实 outcome qualification 未通过；
- Relationship Lab P1k-R1 的 disclosed-policy A 格为 12/12 valid、accuracy 0.50、pair-flip 0，
  判 `substrate_cannot_apply_disclosed_policy`；
- 完整四轴总主张仍关闭。

因此本包不会用外部论文替代仓库 formal，也不会把工程实现存在写成 Volvence 已通过。

## 9. 预注册式研究问题

| ID | 问题 | 支持四轴闭环的结果 | 否定/转向结果 |
|---|---|---|---|
| RQ1 | 命名内部状态是否跨格式稳定？ | CV/J-lens-like readout 在 paraphrase、MC/open、语言变化后保持校准 | 只在同格式可读 → 状态必须标注 domain/format lineage |
| RQ2 | 当前 substrate 是否接受内部控制？ | residual intervention 在 matched budget 下产生 pair-flip 与 N+1 行为效应 | oracle/learned intervention 均无效 → 停止优化 gate，回到 substrate floor |
| RQ3 | PE 是否能训练逐实例 gate？ | PE-gated > static CAST/always-on/noop，heldout worst-seed 通过 | PE 不敏感或方向与 outcome 冲突 → 封存该信用面 |
| RQ4 | Appendable 是否产生真实增益？ | full-loop > memory-only/stateless，且 retention/locality/conflict 同时过门 | naive ICL ≥ memory → 收紧写门，不扩大 CMS |
| RQ5 | 干预副作用能否在应用前预测？ | unsteered state 能稳定预测 cross-effect 方向并降低能力税 | 无可预测性 → promotion 只能依赖逐项运行审计 |
| RQ6 | 后端能否支持可审计 ACTIVE？ | vLLM 路径保留 exact layer/norm/noop/lineage，延迟与 batching 达门 | 需不可审计 fork/hack → 继续 SHADOW |
