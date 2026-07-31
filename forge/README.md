# Volvence RSI Forge

Forge 是一个指向 Volvence 开发环的、物理独立的 RSI 元层。它把已经发生的失败压缩成
可检查的模式，只在人工声明的可编辑面内生成候选 diff，并让验证器与人类决定是否晋升。
它不是新的 runtime wheel，也不是会在在线会话里重写脑核的“超级 owner”。

设计依据包括本仓库归档的 Lilian Weng 文章
[`Harness Engineering for Self-Improvement`](../docs/external/lilian-weng-harness-engineering-2026-07-04.html)
（[来源与哈希](../docs/external/lilian-weng-harness-engineering-2026-07-04.source.md)），以及
Volvence 的 R8/R10/R12/R15 约束。本文只做工程化归纳，不复制原文。

## 为什么现在做 Forge

近程 RSI 不必先从“模型直接改自己权重”开始。部署系统、上下文组织、工具接口、工作流、
持久状态与验证方式共同决定模型能否稳定兑现已有能力。对 Volvence 而言，这一判断有两个
重要含义：

1. Forge 首先是速度和经济杠杆。它缩短“失败发生—找到机制—形成窄改动—验证—沉淀”
   的周期，但不声称凭空提升基座智力。
2. 真正的能力杠杆仍来自更好的 substrate、控制器和训练证据。Forge 不能把 harness 收益
   冒充 NL/ETA 机制收益，也不能绕开 `ModificationGate` 修改产品 runtime。

这解释了它为什么位于 `packages/` 同级的 `forge/`：若放进 `vz-*`，开发流程会变成
runtime owner；若做成完全独立项目，它又失去对本仓库 rules、轨迹与 gate 工件的受控
观察。顶层独立包保留了两者之间清晰的读写边界。

## 优化对象阶梯

Harness 的优化对象可以逐步从指令、结构化上下文、工作流，扩展到 harness code，再扩展到
optimizer code。越向后，搜索空间和收益上限越大，权限风险、归因混淆与回归半径也越大。

Forge 第一阶段刻意停在阶梯前两级：

| 阶段 | 对象 | 当前状态 | 晋升要求 |
|---|---|---|---|
| L1 | `.cursor/rules/*.mdc` 指令与结构化规则 | 开放、append-only proposal | 人审 + held-in/out 验证 |
| L2 | `forge/prompts/**` 分析与提案上下文 | 开放、append-only proposal | 人审 + schema/回归验证 |
| L3 | 工作流定义 | 未开放 | 独立 convergence packet + 新 verifier |
| L4 | `forge/src/**` harness code | 永久在当前循环外 | 第二层 Forge 或人工工程变更 |
| L5 | proposer/optimizer code | 未开放 | STOP 式元优化战役，不能自授权 |
| L6 | 产品 runtime / 模型参数 | 未开放 | `ModificationGate.OFFLINE` + rare-heavy 证据 |

白名单由 [`editable_surface.yaml`](./editable_surface.yaml) 冻结。Forge 不能编辑白名单自身，
也不能通过更宽的 glob 覆盖保护面。

## 与静态 auto-research harness 的区别

AI Scientist 一类系统可以用人工设计的固定流程串起想法、代码、实验、论文与评审；这证明
harness 能协调很长的工作链，但固定流程本身不是 RSI。Forge 的差异是把 harness 的一部分
变成显式优化对象，并形成跨轮闭环：

```text
公开轨迹/证据 → 三层失败记录 → 语义聚类 → 有界 proposal
              → 循环外验证 → 人审 apply → ledger 预测
              → 下一轮真实观测检验预测
```

若只有“根据一次失败手写一条 rule”，这是普通工程修复；若只有“生成很多候选再挑高分”，
但不保留根因、预测和反证，它是不可解释的搜索。Forge 要求每个改动都是一个下一轮可证伪
的文件级主张。

## 三类可观测性如何落地

### 组件可观测性

`editable_surface.yaml` 为每类可编辑组件提供名称、glob 和语义描述。失败模式通过语义嵌入
与实际资产内容对齐；低于阈值就标记 `out-of-surface`，只报告，不强行提案。代码不使用
自然语言关键词把失败映射成 scene/mode/action。

### 经历可观测性

`sources.py` 读取三类公开产物：

- Cursor transcript JSONL：只提取结构化 tool 序列、错误状态与有限错误摘录，不复制完整
  用户对话进 Forge 工件；
- `promotion_verdict.json` 与同目录 `report.md`：读取显式 gate 布尔值、verdict 与报告摘要；
- `.cursor/plans/*.plan.md`：提供战役叙事上下文，永远只读。

`mine.py` 先形成三层记录：终端 verifier 原因、相关 agent 行为因果、暴露的抽象机制；再用
语义嵌入聚类。原始轨迹、记录、pattern 分层保存，避免把所有历史塞回一个不断膨胀的 prompt。

### 决策可观测性

每个 proposal bundle 包含：

- `patch.diff`：尚未落盘的单目标 unified diff；
- `manifesto.json`：证据、根因、目标修复、预测、风险回归、需保留行为与回滚方法；
- `validation.json`：所有循环外检查及其 PASS/BLOCK 理由。

人审决定追加到 [`ledger.jsonl`](./ledger.jsonl)。已应用提案冻结
`pattern_occurrence_count` 的 baseline、方向和下一轮期望 delta；下一次 `mine` 自动给出
`fulfilled / refuted / pending`，而不是把“应用过”当作“有效”。

## 三条硬边界

1. **循环外只读**：packages、tests、gate/verifier 脚本、spec、DATA_CONTRACT、Forge code、
   schema、权限配置与 LLM 配置都不可成为 proposal target。
2. **可编辑面显式白名单**：所有 diff path 必须先通过保护面，再通过 allow glob；第一阶段
   只允许单文件、append-only 候选。
3. **决策可观测**：没有 evidence ref、manifesto、预测、风险与回滚的候选不可验证；没有
   PASS validation 和明确人审者的候选不可 apply。

这些边界让一次收益仍可归因于 harness 编辑，而不是因为候选偷偷关掉测试、换模型、扩大
预算或改 evaluator。

## 验证与 held-in / held-out

Self-improving harness 不能只证明目标失败变好，还要证明未知行为没有退化。第一阶段定义：

- targeted relevance judge = 最小 held-in 检查，判断 diff 是否针对被引用的 failure pattern；
- `forge/tests` = held-in 机制回归；
- `tests/contracts/test_forge_boundaries.py` = held-out 边界回归；
- Ruff 与 patch clean-apply = 静态准入。

验证器从 `editable_surface.yaml` 读取冻结命令。提案不能修改命令、测试、judge prompt 或 schema。
任一检查缺失、超时、不可执行或不通过都得到 `BLOCK`；没有“先应用再看看”的静默回退。

第一阶段的局限也必须诚实：规则编辑尚没有大规模任务级 held-out benchmark，因此 PASS 只表示
结构、相关性和边界检查通过，不等于已证明真实 pass rate 提升。ledger 的下一轮预测兑现才是
最早的纵向证据；扩大写面前还需要独立 benchmark split。

## 域特有风险

### 模糊 evaluator

关系质量、研究品味和长期可维护性没有快速硬 verifier。Forge 因此从开发环开始：lint、测试、
gate verdict 和 patch apply 都是较硬的信号。遇到只靠 LLM judge 的结论必须保留
`out-of-surface` 或 BLOCK，不把 judge 分数反向变成 Volvence 的学习源。

### 多样性坍缩

重复出现的高分改法可能把所有 rules 推向同一种冗长模板。`propose.py` 使用与 failure mining
相同的语义 embedding 空间，对本轮候选和 ledger 历史补丁做余弦去重；相似度过高就拒绝。
后续若演化候选池，还需 Pareto 维度覆盖可维护性、范围与验证成本，而非只看单一 pass rate。

### Reward hacking

最直接的作弊路径是删测试、改阈值、换 judge、扩大 token/时间预算或改权限。保护面优先级、
单目标 diff、preimage hash、clean-apply、固定验证命令和人审共同阻断这些路径。验证报告与
manifesto 都做内容哈希，apply 前再次核对，防止 validate 后替换候选。

### 长期仓库健康

短期任务成功可能换来 ownership 侵蚀、兼容债与调试成本。proposal 必须列出 preserve behavior
和 at-risk regressions；append-only 是第一阶段的保守限制，不是长期最佳编辑策略。只有在积累
足够 negative result 与 rollback drill 后，才考虑允许结构化替换。

## 两阶段路线

### 战役一：开发环 Forge（本目录）

- 读公开 transcript、verdict、report、plan；
- 编辑 rules 与 Forge prompts；
- 由固定测试、judge 和人审晋升；
- 不 import `volvence_zero.*` 或 `lifeform_*`；
- 不写 runtime slot，不更新模型权重。

### 战役二：产品 runtime Forge（未实现）

未来候选面可能包括 reviewed playbook、角色包或记忆整合策略，但必须先在正式 owner 契约注册，
proposal 经 `ModificationGate.OFFLINE`，并以 `DISABLED → SHADOW → ACTIVE` 迁移。开发环 ledger
不能直接充当 runtime credit，也不能让 evaluation 反向成为 PE 源。

## 使用

Forge 不进入根 workspace：

```bash
python -m pip install -e 'forge[dev]'
```

挖掘真实输入：

```bash
forge mine
```

生成候选（需要显式 OpenAI-compatible 环境配置，或测试/演练用 replay backend）：

```bash
forge propose artifacts/forge_mine_<timestamp>/failure_patterns.jsonl
```

验证不会修改目标文件：

```bash
forge validate artifacts/forge_propose_<timestamp>/proposals/<proposal_id>
```

只有人类明确批准后才能落盘：

```bash
forge apply artifacts/forge_propose_<timestamp>/proposals/<proposal_id> \
  --validation-report artifacts/forge_propose_<timestamp>/proposals/<proposal_id>/validation.json \
  --human-approved-by '<reviewer>'
```

拒绝也写 ledger：

```bash
forge apply <proposal_dir> --reject --reason '<reason>' --human-approved-by '<reviewer>'
```

## 回滚与退出条件

Forge 自身全部是新增目录；移除 `forge/`、对应 spec 与 contract test 即可回滚，不影响 runtime。
已 apply、尚未提交的 rule 改动在仓库根目录执行 manifesto 中记录的精确
`git apply --reverse <repo-relative-patch>` 命令回滚；提交后的改动按仓库纪律用独立 revert commit 回滚，ledger
保留历史，不删除负结果。

战役一退出条件：真实 93 份 transcript 与至少一份失败 gate bundle 能完成
`mine → propose → validate`，产出至少一个可人审 bundle；边界测试、Forge tests 与 Ruff 全部
通过。任何写面越界、验证器可编辑、预测无法复核或 held-out 回归都立即 BLOCK 并保持现状。
