# NVIDIA AVO 架构专项研究

> 日期：2026-08-29
>
> 对象：NVIDIA `Agentic Variation Operators`（AVO）
>
> 状态：研究结论；不构成 spec、实现、研究启动或 runtime wiring 授权

## 一句话裁决

AVO 不是一种新的基础模型架构，也不是 Volvence 意义上的在线持续学习器。它把一个具备计划、工具、
执行反馈、持久上下文和监督机制的通用 coding agent，提升为进化搜索中的**整个变异算子**：代理自行决定
读什么、选哪段谱系、改什么、何时测试、失败后如何修复，直到提交一个正确且不劣于当前最优的候选。

它与 Volvence 最强的交点位于 **Forge / Praxist 的离线研究面**，而不是 `vz-memory`、PE→credit、
`z_t` Internal RL 或 residual steering 的运行时主链。最合理的吸收方式是：让 AVO-style agent 在一个
owner-scoped、evaluator 可验证、权限受限的研究沙箱里承担 variation operator，同时完整保留 Volvence
现有的 A0 人审、loop-external validation、`ModificationGate` 与
`DISABLED → SHADOW → ACTIVE`。AVO 的 autonomy 应增强“怎样找候选”，不能获得“什么是真理”或
“谁能上线”的权力。

## 核心事实

- 对应原论文是 Chen et al., **AVO: Agentic Variation Operators for Autonomous Evolutionary Search**，
  arXiv [`2603.24517`](https://arxiv.org/abs/2603.24517)，2026-03-25 v1。
- 论文正式实验只覆盖 NVIDIA B200 上的 attention kernel 搜索：7 天、超过 500 个内部方向、40 个
  committed versions；在论文测试配置上，MHA 相对 cuDNN 最高 `+3.5%`、相对 FlashAttention-4
  最高 `+10.5%`。
- 论文实验是 **single-lineage continuous run**，不是 population / island / MAP-Elites 实验；
  population-level branching 被明确留作未来工作。
- 论文中的持久记忆主要描述为累积的 conversation history、实现、编译器/Profiler 输出和 reasoning，
  committed lineage 以 git commit 保存；它没有给出类型化、多时间尺度、删除/衰减或 owner snapshot 契约。
- 2026-08-21 NVIDIA 技术博客把同一 harness 扩展到 ARC-AGI-3 public set，报告 Claude Opus 5
  完成 25 个公开环境、183 个 level、`100.00 RHAE`、6,624 次环境动作。该结果不是 AVO 原论文的一部分，
  也不是 private/semi-private set 结果。
- NVIDIA 明确承认 AVO 与 VISTA 的动作数比较不是受控消融；memory、supervisor、observation representation
  或 context management 的单独贡献都没有被隔离。
- 截至本研究日期，AVO 论文与 NVIDIA 官方博客没有链接官方代码、完整运行轨迹、agent prompt、确切
  kernel-run 模型配置或可重放 artifact。第三方复现不作为 NVIDIA 实现证据。

## 与 Volvence 的总关系

| 问题 | AVO | Volvence | 裁决 |
|---|---|---|---|
| 优化对象 | 外部代码 / 工程 artifact | runtime state、controller、memory，以及受治理的 rare artifact | 不同层级 |
| 适应机制 | frozen/unspecified frontier model 的长程 in-context agent search | PE→credit 的有界 owner-local 学习；rare-heavy 另走 Gate | 不能互换 |
| 记忆 | conversation/context + past implementations/results + git lineage | CMS 多时间尺度 + semantic owners + frozen snapshots/hydration | AVO 只提供工程启发 |
| evaluator | `f` 直接驱动搜索与 commit | development evaluator 可调度研究；runtime evaluation 禁止回灌学习 | 只可放研究面 |
| 自主性 | agent 自主合并 Sample/Generate/Evaluate | 研究自治与 validation/admission/deployment 分权 | 可借 operator，不借 authority |
| 发布 | 正确且不劣即进入 committed lineage | retention → handoff → external validation → Gate → SHADOW → canary → ACTIVE | VZ 治理更完整 |
| 实证 | 7 天真实 B200 搜索；public ARC 满分 | 四轴机制与 SHADOW 证据丰富，但新的自动研究控制链尚无真实 run | AVO 运行证据更强 |

## 最值得借鉴的五点

1. 把 agent 从“一次生成候选”升级成**可自行读文档、诊断、修复、反复评测的 variation operator**。
2. 把 `lineage + domain knowledge + executable evaluator + tools` 冻结成任务接口，使同一个 agent loop
   能换域而不重写 agent 本体。
3. 为 plateau / repeated failure 增加独立 supervisor，但 supervisor 只发布重定向建议，不能改 evaluator、
   Gate、protected roots 或部署权限。
4. 用长时运行、恢复、失败循环与单位预算进展衡量 harness，而不是只看单次候选质量。
5. 把模型能力与 harness 能力分开评估；对 memory、supervisor、lineage 和 context policy 做 matched
   ablation，补上 AVO 当前没有完成的归因。

## 明确不借鉴

- 不把 conversation history 当作 CMS 已成立。
- 不把 benchmark score、LLM judge 或 research frontier rank 写入 runtime PE/credit。
- 不让同一个 agent 同时拥有候选生成、正式验证、Gate 与 production apply。
- 不把 best-so-far 单谱系用于关系、人格、边界等没有单一 hard verifier 的领域。
- 不从 public ARC-AGI-3 饱和结果推出 unseen generality、持续学习或 AGI 主张。
- 不开放 LLM 任意修改 owner 源码、evaluator、schema、权限或 wiring。

## 建议阅读顺序

1. [`01_AVO_ARCHITECTURE_AND_EVIDENCE.md`](./01_AVO_ARCHITECTURE_AND_EVIDENCE.md)：论文架构、公式、
   kernel 实验、ARC 扩展与证据缺口。
2. [`02_VOLVENCE_COMPARISON.md`](./02_VOLVENCE_COMPARISON.md)：逐层对比、四能力轴对账与项目当前状态。
3. [`03_ADOPTION_RECOMMENDATIONS.md`](./03_ADOPTION_RECOMMENDATIONS.md)：适合当前仓库的最小试验包、
   ablation、指标、kill condition 和回滚边界。
4. [`SOURCES_AND_DOWNLOADS.md`](./SOURCES_AND_DOWNLOADS.md)：原始来源、相邻论文、开源与下载审计。

## 与仓库已有研究的关系

AVO 是 [`AlphaEvolve 专题`](../probe/notes/cross-axis/alphaevolve-evolutionary-borrow-2026-05.md)
的增量，而不是替代：

- AlphaEvolve 的强项是 archive、island、MAP-Elites 风格多样性、evaluation cascade 与多目标搜索；
- AVO 的新增点是把 `Sample + Generate + inner evaluation/repair` 交给一个长程、自主、可用工具的 agent；
- AVO 当前实验反而退回 single-lineage，因此 Volvence 不应为了引入 agentic operator 丢掉已经采用的
  Pareto / Frontier / QD 思路。

本研究没有修改任何 runtime owner、DATA_CONTRACT、ResearchTask schema 或 Praxist 配置，也没有启动
研究任务。它只提出一个下一步可证伪的 task-local pilot 设计。
