# LLM 内部控制与持续深度学习：四能力轴缺口研究（2026-08）

> Status: 大型研究项目 / 外部证据合成，**不是 runtime contract，不授权 ACTIVE**。
> As of: 2026-08-22。
> 种子判断：[`../continual-deep-learning.md`](../continual-deep-learning.md)。本包保留种子原文，
> 在其上增加证据分级、缺口排序、可复用机制、反例、工程路线和可证伪实验。

## 核心裁决

公开证据已经足以支持一个较窄但重要的结论：

> 现代 LLM 的内部激活中存在可读、可写、具有因果作用的控制表征；其中一部分甚至会被模型
> 自己按任务要求调入并用于推理。

但这不等于“LLM 已经会持续、自主、安全地控制自己”。截至本项目截止日，在检索到的公开
一手材料中，尚无系统同时证明：

```text
跨 session 追加
  → 命名且校准地读出
  → 只由行动后的 Prediction Error / credit 学习
  → 逐实例、有界、可择时地干预冻结基底
  → 改变真实下一拍并结算
  → 可删除、可恢复、可单字段回滚
```

业界已经分别推进到以下位置：

| 层级 | 最强公开证据 | 当前上限 |
|---|---|---|
| 内生控制面 | Anthropic J-space | 模型能调入、复用和因果使用少量可言说概念；仍是单次前向内的工作空间 |
| 表征执行器 | Function Vectors、ReFT、Refusal direction | 冻结基底上能用紧凑表征改变任务或行为；可靠性高度依赖模型、格式、层和样本 |
| 条件与动态控制 | CAST、FASB、TACT、逐实例多层 steering | 已从 always-on 走到条件门、轨迹监控、逐实例选层和自适应剂量；门多数仍由离线标签、阈值或启发式构造 |
| 反馈控制 | PID Steering、Activation-LQR | 已把 steering 写成反馈控制问题并给出局部稳定/误差分析；尚未等价于真实 outcome 的在线信用学习 |
| 可追加神经状态 | Titans、ATLAS、Nested Learning / HOPE、TTT-E2E | 测试时写神经记忆或快权重已可扩到超长上下文；精确记忆、冲突、删除和跨 session 主体连续性未解决 |
| 学习控制器 | ETA Internal RL、SEAL | 潜在 controller 与自编辑策略可以学习；前者停留在仿真外部奖励，后者修改权重并有遗忘与成本问题 |
| 部署基础设施 | pyReFT、IBM activation-steering、vllm-lens、Goodfire | 捕获与注入 activation 已具工程可行性；公开材料没有完整长期 SLA、能力税、回滚和用户价值证据 |

所以 Volvence 的潜在独特性不再是“发现内部可以控制”，而是：

> 把内部控制面纳入 Appendable → Readable → Learnable → Steerable 的有界、可审计、
> PE-only、跨 session 闭环，并在真实下一拍上证明它比 memory-only、static steering 和
> stateless 三类强基线都更好。

## 当前最重要的五个未解问题

1. **控制权是否真的到达目标基底。** Relationship Lab P1k-R1 已披露策略仍只有
   `0.50` accuracy、`0` pair-flip；当前首要瓶颈是 substrate/readout floor，不是缺一个更聪明的 gate。
2. **读出的轴是否可迁移且可干预。** “可解码”不推出“可扳动”；即便有因果作用，也不推出跨格式、
   跨语言、跨模型或跨版本不变。
3. **N+1 PE 是否对用户可见行为敏感。** 当前 C3 信用面首先衡量表示对齐，不自动等于行为改变或用户侧结果。
4. **记忆写入是否比不写更好。** CL-Bench 中 naive ICL 胜过专用记忆系统；错误泛化、陈旧信念和冲突写入
   是 Appendable 的核心风险，不是边缘问题。
5. **逐实例干预能否长期无副作用。** 固定方向、固定层和过量 steering 会反向伤害已有正确样本；
   side-effect matrix、capability tax、strict noop 和 rollback 必须成为 promotion gate。

完整排序与每项 kill condition 见 [`03_UNSOLVED_PROBLEMS.md`](03_UNSOLVED_PROBLEMS.md)。

## 谁可以被我们所用

| 外部工作 / 工具 | 可用部分 | 在 Volvence 中的角色 | 决策 |
|---|---|---|---|
| J-space / Jacobian lens | 因果命名读出、workspace 选择性、概念 swap/ablation | Readable 候选仪器与“内生控制面”对照 | **原型验证** |
| Concept Vectors（ICLR 2026） | 用跨格式 RSA 选稳定 heads | 修复 Function Vector 的格式绑定 | **优先借鉴** |
| steering reliability 系列 | direction coherence、class separation、reverse-effect 检查 | 所有 executor 训练前的 steerability screen | **直接采用判据** |
| ReFT / pyReFT | 冻结基底、低秩学习式表征干预 | executor 训练与 artifact 格式参考 | **适配后采用** |
| CAST | 条件 gate 与 behavior vector 解耦 | static-gate 强基线、gate-off 对照 | **直接作为基线** |
| TACT | 轨迹内读漂移、只在必要 step 修正、真实终局指标 | coding-lab 外部同构与长轨迹评测模板 | **优先复现思想** |
| 逐实例多层 steering | prompt-conditioned 选层、方向预测、adaptive-K gate | 逐 turn layer/dose scheduler | **最高价值新借鉴** |
| Forecasting Side Effects | 67 行为 cross-effect matrix、干预前副作用预测 | promotion 前的 capability-tax / spillover audit | **最高价值新借鉴** |
| FASB / PID / Activation-LQR | 动态强度、backtracking、反馈 setpoint | controller 设计备选，不替代 PE owner | **研究性适配** |
| Titans / HOPE | surprise 驱动写入、多时间尺度更新 | memory write salience 与频率调度先验 | **只借机制，不借主张** |
| TTT-E2E | 当前输入局部更新、压缩记忆的精确检索反例 | CMS 分层与 exact-memory 保底依据 | **借边界与负结果** |
| CL-Bench | gain metric、stateful real-domain protocol | 四臂纵向实验与 headroom 归一 | **直接采用评测思想** |
| vllm-lens / Goodfire | inference server 内捕获、注入和高吞吐 harvest | vLLM 可行性 spike | **隔离评估，不直接绑主链** |

逐项“借什么 / 改什么 / 禁止照搬 / owner 落点”见
[`04_REUSABLE_MECHANISMS.md`](04_REUSABLE_MECHANISMS.md)。

## 文档结构与阅读顺序

| 文件 | 解决的问题 |
|---|---|
| [`00_RESEARCH_CHARTER.md`](00_RESEARCH_CHARTER.md) | 研究问题、术语、证据等级、检索和诚实边界 |
| [`01_FIELD_MAP.md`](01_FIELD_MAP.md) | 2024–2026 内部控制、steering、持续记忆、Internal RL 与工程路线全景 |
| [`02_FOUR_AXIS_CROSSWALK.md`](02_FOUR_AXIS_CROSSWALK.md) | 逐工作映射 Appendable / Readable / Learnable / Steerable，防止局部结果冒充闭环 |
| [`03_UNSOLVED_PROBLEMS.md`](03_UNSOLVED_PROBLEMS.md) | Volvence 尚未解决的问题、外部证据、唯一 owner、决定性实验与 kill condition |
| [`04_REUSABLE_MECHANISMS.md`](04_REUSABLE_MECHANISMS.md) | 可直接借、需改造、只作基线、明确拒绝的机制清单 |
| [`05_EVIDENCE_ROADMAP.md`](05_EVIDENCE_ROADMAP.md) | 七个收敛包、四臂主实验、晋升门、顺序与退出条件 |
| [`06_ENGINEERING_FEASIBILITY.md`](06_ENGINEERING_FEASIBILITY.md) | hooks、batching、延迟、artifact lineage、后端和回滚的工程判断 |
| [`07_NEGATIVE_RESULTS_AND_RISKS.md`](07_NEGATIVE_RESULTS_AND_RISKS.md) | 反向 steering、格式不变性、能力税、记忆污染、精确检索和 evaluator 泄漏 |
| [`08_SOURCE_LEDGER.md`](08_SOURCE_LEDGER.md) | 一手来源、发表状态、精确主张、证据等级和本地相邻研究入口 |
| [`09_DECISION_REGISTER.md`](09_DECISION_REGISTER.md) | 当前 Adopt / Adapt / Baseline / Watch / Reject 决策与复审触发器 |

三条快捷阅读路径：

- **只看战略判断**：本页 → `03` → `09`。
- **准备下一轮实验**：`02` → `03` → `04` → `05`。
- **准备工程 spike**：`04` → `06` → `07` → `08`。

## 与已有研究包的边界

- [`../continual-learning-2026-07/`](../continual-learning-2026-07/)：持续学习七派、CL-Bench、
  Spurious Forgetting、个人参数化的逐篇深读；本包引用其结论，不复制 25+22 篇笔记。
- [`../steering-2026-08/`](../steering-2026-08/)：S2 null、steerability precheck、C2 条件执行器、
  S3-E 择时学习的仓库内实验；本包把它放入更大的外部证据图。
- [`../ttt-e2e-long-context-2026-08/`](../ttt-e2e-long-context-2026-08/)：TTT-E2E 专篇；本包只消费其
  “压缩记忆不能代替精确记忆”裁决。
- [`../anthropic-emotion-concepts-2026-04/`](../anthropic-emotion-concepts-2026-04/)：emotion concepts
  专篇；本包强调“局部 operative concept 不等于持续 self state”。

## 研究结论的使用限制

- `D` 只表示论文在自己的设置内给出直接证据，不表示满足 Volvence 的 owner、snapshot、PE-only、
  rollback 或 production 约束。
- arXiv / 公司研究报告与同行评审论文分开标记；工程吞吐自报不当作行为有效性证据。
- 自动 judge、人工标签和 benchmark 分数只能做验证或外部 outcome；不得因此进入 PE 学习源。
- 本包不修改任何 runtime owner、slot、WiringLevel 或 production 默认。
