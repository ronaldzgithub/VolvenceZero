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

Forge 的生产写面仍高度收窄：开发环资产之外只开放一个真实消费的 companion additive
playbook overlay。它默认 `DISABLED`，apply 与 ACTIVE 部署是两个独立决定：

| 阶段 | 对象 | 当前状态 | 晋升要求 |
|---|---|---|---|
| L1 | `.cursor/rules/*.mdc` 指令与结构化规则 | 开放、append-only proposal | 人审 + held-in/out 验证 |
| L2 | `forge/prompts/**` 分析与提案上下文 | 开放、append-only proposal | 人审 + schema/回归验证 |
| L3 | 工作流定义 | 未开放 | 独立 convergence packet + 新 verifier |
| L4 | `forge/src/**` harness code | 永久在当前循环外 | 第二层 Forge 或人工工程变更 |
| L5 | proposer/optimizer code | 循环外实现 Pareto/STOP，提案不可编辑 | 新指标或自演化需独立人工战役 |
| L6a | companion runtime overlay | 精确单文件候选面开放，live 默认 DISABLED | owner validator + frozen suite + OFFLINE ALLOW + 人审；另行部署 wiring |
| L6b | 模型参数 | 只开放 DISABLED rare-heavy build request | substrate train/evaluate + cognition gate + loop-external READY；publish 仍独立 |

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

`sources.py` 读取五类公开产物：

- Cursor transcript JSONL：只提取结构化 tool 序列、错误状态与有限错误摘录，不复制完整
  用户对话进 Forge 工件；
- `promotion_verdict.json` 与同目录 `report.md`：读取显式 gate 布尔值、verdict 与报告摘要；
- `.cursor/plans/*.plan.md`：提供战役叙事上下文，永远只读。
- Companion Bench `*.bundle.json` / `arc_failure.jsonl`：提取逐轮 rubric、disqualifier、arc axis
  与显式 transport/runtime failure。bench provenance 与开发轨迹分 lane，不能映射到 rules/prompts。
- 显式提供的 `lifeform-live-dialogue-outcome.v1`：复核 content hash 后只读取去标识化 typed outcome
  和无文本 action context。Forge 不自动扫描 closed-alpha 隐私目录，也不把任意 outcome 硬编码为失败。

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

1. **循环外只读**：除精确 companion overlay JSON 外，packages、tests、gate/verifier 脚本、
   spec、DATA_CONTRACT、Forge code、schema、权限配置与 LLM 配置都不可成为 proposal target。
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

冻结 synthetic task-decision split 可以独立检查当前 asset，也可以比较候选：

```bash
forge benchmark .cursor/rules/cursor-convergence-workflow.mdc \
  --backend openai

forge benchmark .cursor/rules/cursor-convergence-workflow.mdc \
  --candidate-asset /tmp/candidate.mdc \
  --backend openai
```

suite、benchmark prompt 与 schema 都在 proposal 循环的只读面。模型输入不含 case 的 expected、
critical 或置信阈值；报告按 exact decision、minimum confidence、critical failure、绝对 pass rate 与
candidate delta 判定 `PASS/BLOCK`。报告始终标记 `diagnostic_only=true` 和
`causal_claim_authorized=false`，不会自动接入 validate/apply gate。

局限仍必须诚实：这组冻结 case 只覆盖结构化 task decision，不执行真实仓库任务，也没有独立
verifier 形成 causal outcome。因而 PASS 不等于已证明真实 pass rate 提升；ledger 的下一轮预测兑现
仍只是最早的纵向证据，扩大写面前还需要真实任务 split 与多次独立运行。

## 域特有风险

### 模糊 evaluator

关系质量、研究品味和长期可维护性没有快速硬 verifier。Forge 因此从开发环开始：lint、测试、
gate verdict 和 patch apply 都是较硬的信号。遇到只靠 LLM judge 的结论必须保留
`out-of-surface` 或 BLOCK，不把 judge 分数反向变成 Volvence 的学习源。

### 多样性坍缩

重复出现的高分改法可能把所有 rules 推向同一种冗长模板。`propose.py` 使用与 failure mining
相同的语义 embedding 空间，对本轮候选和 ledger 历史补丁做余弦去重；相似度过高就拒绝。
`--candidates-per-pattern` 可形成有界候选池；`forge select` 同时考虑 validation delta、capacity
cost、added lines 与 risk count，并在无 eligible candidate 时发布 STOP。它仍是冻结的工程
selector，不声称已经学会优化器，也不自动 apply。

### Reward hacking

最直接的作弊路径是删测试、改阈值、换 judge、扩大 token/时间预算或改权限。保护面优先级、
单目标 diff、preimage hash、clean-apply、固定验证命令和人审共同阻断这些路径。验证报告与
manifesto 都做内容哈希，apply 前再次核对，防止 validate 后替换候选。

### 长期仓库健康

短期任务成功可能换来 ownership 侵蚀、兼容债与调试成本。proposal 必须列出 preserve behavior
和 at-risk regressions；append-only 是第一阶段的保守限制，不是长期最佳编辑策略。只有在积累
足够 negative result 与 rollback drill 后，才考虑允许结构化替换。

## 战役一与第三至第五阶段

### 战役一：开发环 Forge（本目录）

- 读公开 transcript、verdict、report、plan；
- 编辑 rules 与 Forge prompts；
- 由固定测试、judge 和人审晋升；
- 不 import `volvence_zero.*` 或 `lifeform_*`；
- 不写 runtime slot，不更新模型权重。

### 阶段三：唯一 runtime overlay

`lifeform-domain-emogpt` 的 companion package 是真实 runtime consumer。Forge 只允许向
`runtime_assets/companion_playbook_overlay.json` 的 `/playbook_rules` 追加结构化 rule；owner loader
禁止替换 baseline id/pattern，并按外部 `DISABLED/SHADOW/ACTIVE` 编译到既有
`DomainExperiencePackage → strategy_playbook` 通道。生产 asset 初始为空、builder 默认 DISABLED。

### 阶段四：population、Pareto 与 STOP

同一 failure pattern 可生成多个 bounded proposal。`forge select` 重算 proposal/validation/gate
哈希，只从 PASS + 所需 OFFLINE ALLOW 候选中取 component-specific Pareto front；空集必须 STOP。

### 阶段五：rare-heavy 请求，不是权重自写

`forge plan-rare-heavy` 冻结模型、数据、control basis、held-out 与全部超参数，只写 DISABLED
request。训练和评估仍由既有 `train_common_adapter_model.py` 执行，cognition gate 仍是唯一
ALLOW owner。外部 adjudicator 只发布 READY/STOP，不创建或激活 `CommonAdapterBundle`。

## 使用

Forge 不进入根 workspace：

```bash
python -m pip install -e 'forge[dev]'
```

挖掘真实输入：

```bash
forge mine
```

只读取 applied event 之后的新证据，并接入 Companion Bench：

```bash
forge mine --bench-root artifacts --evidence-since-ledger
```

显式接入 opt-in closed-alpha typed outcome（目录不会被自动发现）：

```bash
forge mine \
  --live-outcome-root /path/to/evidence/live_dialogue_outcomes \
  --evidence-since-ledger
```

生成候选（需要显式 OpenAI-compatible 环境配置，或测试/演练用 replay backend）：

```bash
forge propose artifacts/forge_mine_<timestamp>/failure_patterns.jsonl \
  --candidates-per-pattern 3
```

验证不会修改目标文件：

```bash
forge validate artifacts/forge_propose_<timestamp>/proposals/<proposal_id>
```

若目标属于 `requires_offline_gate` component（当前只有 companion overlay），还必须先在循环外裁决：

```bash
python scripts/forge_gate_adjudicator.py \
  artifacts/forge_propose_<timestamp>/proposals/<proposal_id>
```

对一个已验证 population 选择 Pareto front；没有合格候选时命令以 STOP/非零退出：

```bash
forge select artifacts/forge_propose_<timestamp>/proposals
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

第五阶段只规划 rare-heavy 构建请求（所有输入必须已经冻结并可计算 SHA-256）：

```bash
forge plan-rare-heavy \
  --model-id '<model-id>' \
  --model-weights-sha256 '<64-hex>' \
  --common-adapter-version '<version>' \
  --traces '<traces.jsonl>' \
  --control-basis '<control-basis.json>' \
  --held-out '<held-out.jsonl>' \
  --hook-layers '10,11,12'
```

真实训练、held-out evaluate 与 cognition gate 完成后，可做只读绑定裁决：

```bash
python scripts/forge_common_adapter_adjudicator.py \
  --request '<request.json>' \
  --candidate '<common-adapter-candidate.json>' \
  --evaluation-report '<evaluation.json>' \
  --gate-record '<gate.json>' \
  --held-out '<held-out.jsonl>' \
  --output '<artifacts/.../verdict.json>'
```

`READY` 之后仍需单独执行 substrate 的 publish 流程；Forge 不替用户执行该动作。

### 类型化研究机会发现

`forge mine` 产出的 `forge-failure-pattern.v3` 可以进入一个有界扫描回合：

```bash
forge research-scan 'artifacts/forge_mine_<timestamp>/failure_patterns.jsonl' \
  --registry 'forge/research_task_registry.yaml' \
  --once \
  --json
```

scanner 会为每条合法 record 写 content-addressed `ResearchOpportunity`，并且只按 registry 中的 exact
`editable_component / editable_target` 映射任务。因果 prose、标题、excerpt 和字符串相似性均不参与
路由。无映射或 out-of-surface 的机会保留为 `NEEDS_TASK_DESIGN`；匹配项最多按 registry 的
`max_new_requests_per_scan` 提交新 Request，超额项写 `DEFERRED_BY_SCAN_LIMIT`。

当前 registry 只登记一个 exact `coding_memory_inheritance` pilot；登记本身不会创建真实研究预算。
每条任务必须同时冻结 Volvence ResearchTask、Praxist task project、executable、run root、exact
model/profile 与人工维护的 `binding_revision`。scanner 只提交 Request，不批准、不调用 Praxist，也不改变
runtime wiring。完整契约见
[`research-opportunity-discovery.md`](../docs/specs/research-opportunity-discovery.md)。

当前 pilot 已生成一个 `AWAITING_RESEARCH_APPROVAL` Request；它不是批准。拥有主机 shell 权限的其他
终端仍可绕过 Forge 直接调用 Praxist，此类 run 因 id/dir/Approval 不匹配而必须停止并排除，不能冒充
控制面启动的研究。

### A0 自动研究启动控制面

需求驱动发现与研究调度共用 `forge` owner。人类先把完整 Volvence Demand draft 封存到 inbox；之后每次
`research-loop --once` 只做有上限、可重放的一次 pass：

```bash
forge research-demand-seal 'research/demand_drafts/<name>.json' --json

forge research-loop --once \
  --backend codex_sdk \
  --model gpt-5.6-luna \
  --json
```

第一轮只产生 `UNBOUND` TopicProposal。具名人类通过 `research-bind-topic` 绑定 exact mapping 后，下一轮只提交
ResearchRequest；另一次 A0 人审批准后，后续 pass 才调用既有 targeted reconcile。未变化的 pass 是零模型调用、
零新 Request、零新 run；`RUN_COMPLETED` 后不会自动 import Candidate 或上架。

仓库根 `start_research_lab.sh` 在 controlled mode 默认周期唤醒该 bounded loop；它不是新 lifecycle owner，
`--no-auto-research` 可关闭。也可以不启动 Portal，由任意外部 scheduler 调用同一个 `forge research-loop --once`。
Praxist 自己托管 detached run。显式控制面入口仍可单独使用：

```bash
forge research-submit '<task.json>' \
  --task-project '<praxist-task-project>' \
  --praxist-executable '<absolute-praxist-executable>' \
  --run-dir '<absolute-new-run-dir>' \
  --requested-by '<detector-or-human>' \
  --reason '<typed-research-reason>' \
  --agent-system '<claude_sdk-or-codex_sdk>' \
  --runtime '<agent_runtime:...>' \
  --model-provider '<model_provider:...>' \
  --model '<exact-model>' \
  --cohort 4 \
  --generations 8

forge research-inbox

forge research-approve '<request.json>' \
  --approved-by '<human>' \
  --reason '<approval-reason>'

forge research-reconcile --once
```

Request 会冻结 Volvence Task、Praxist task project、executable/source identity、run dir、model/profile
与预算；批准后的调和顺序固定为 active-capacity check → doctor → resolve →
`start --daemonize --json` → targeted status。`START_INTENT` 先于 launch create-only 落盘，worker
中断后先按 exact `run_id` 恢复，禁止重复 start。命令不保存 provider credential 或 raw stderr。
`--codex-native` 的 doctor/resolve/start 还会清除 API key、base URL、binary 与 model 环境覆盖，只使用
当前主机保存的 ChatGPT 登录；非 native profile 的显式 provider 凭据行为不变。

`research-submit` 是 typed scanner 与显式人工提交共用的 seam；scanner 也不会从自然语言或关键词
发现问题。没有 runnable task project 的机会不会自动启动。`RUN_COMPLETED` 只结束研究生命周期，
仍需下面的 Handoff、loop-external validation、ModificationGate 和两级部署审批。完整契约见
[`research-control-plane.md`](../docs/specs/research-control-plane.md)。

### 通用研究候选上架

所有 Praxist 研究任务可通过同一离线桥接面交付候选。先验证 Volvence-owned Task，再从一个
committed generation boundary 导入候选：

```bash
forge research-validate-task '<task.json>'
forge research-import-praxist '<task.json>' '<handoff.json>' --run-dir '<praxist-run>'
```

独立 validator 与正式 `ModificationGate` adapter 发布 exact-bound JSON 后，Forge 只签发授权收据：

```bash
forge research-authorize '<task.json>' '<candidate.json>' '<validation.json>' '<gate.json>' \
  --to-wiring shadow \
  --authorized-by '<reviewer>' \
  --reason '<reason>'
```

`ACTIVE` 必须额外提供紧邻的 AUTHORIZED SHADOW receipt。回滚只依赖上一张授权收据，避免因
研究目录或收益证据不可用而阻止降权：

```bash
forge research-rollback '<previous-receipt.json>' \
  --to-wiring disabled \
  --authorized-by '<operator>' \
  --reason '<reason>'
```

这些命令均不会 apply candidate、调用 gate 或修改 runtime wiring；目标 owner 仍须消费 receipt，
执行自己的 SHADOW/canary/ACTIVE 协议。完整契约见
[`research-promotion-pipeline.md`](../docs/specs/research-promotion-pipeline.md)。

## 回滚与退出条件

Forge 自身可独立移除；companion overlay 的运行时回滚优先把 wiring 设为 `DISABLED`，若已经
apply 内容，再执行 proposal 记录的 reverse patch 或恢复前一 overlay。rare-heavy request/verdict
未发布时可直接停止使用，不影响 runtime。
已 apply、尚未提交的 rule 改动在仓库根目录执行 manifesto 中记录的精确
`git apply --reverse <repo-relative-patch>` 命令回滚；提交后的改动按仓库纪律用独立 revert commit 回滚，ledger
保留历史，不删除负结果。

战役一退出条件：真实 93 份 transcript 与至少一份失败 gate bundle 能完成
`mine → propose → validate`，产出至少一个可人审 bundle；边界测试、Forge tests 与 Ruff 全部
通过。任何写面越界、验证器可编辑、预测无法复核或 held-out 回归都立即 BLOCK 并保持现状。
