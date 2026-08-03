# RSI Forge Spec

> Status: phase 3–5 contracts landed；唯一 runtime overlay 默认 DISABLED；无 live/GPU promotion
> Last updated: 2026-08-03
> 对应需求: R8, R10, R12, R15

## 要解决的问题

如何从开发代理的真实失败中形成可归因、可证伪、可回滚的 harness 改进循环，同时不让生成
提案的系统修改 evaluator、权限边界、runtime owner 或自身优化器？

## 能力定位与唯一 owner

`volvence_forge` 是开发环 proposal/evidence owner，只拥有以下离线工件：

- 解析后的公开 source record；
- 三层 failure record 与语义聚类后的 failure pattern；
- 未落盘的 proposal bundle；
- 人审 decision ledger 与下一轮 prediction check。

它不是 `vz-*`/`lifeform-*` wheel，不注册 `docs/DATA_CONTRACT.md` runtime slot，不发布 Brain
snapshot，也不成为 PE、credit、evaluation、gate、memory 或 temporal 的第二 owner。

## 输入契约

| 输入 | 读取方式 | 权限 | 说明 |
|---|---|---|---|
| Cursor transcript `*.jsonl` | 结构化逐行解析 | 只读 | tool 序列、显式错误、turn status；不复制完整用户对话到工件 |
| `promotion_verdict.json` | JSON + 显式布尔 gate leaf | 只读 | 同目录 `report.md` 只提供有限摘要 |
| Companion Bench `*.bundle.json` / `arc_failure.jsonl` | 结构化解析 | 只读 | 逐轮 rubric、disqualifier、arc axis 与 transport/runtime failure；不把 judge 变成学习 owner |
| lifeform-service `lifeform-live-dialogue-outcome.v1` | exact JSON + content hash | 显式路径、只读 | closed-alpha typed outcome 与无文本 action context；不自动发现隐私目录、不由代码判定 failure |
| `.cursor/plans/*.plan.md` | YAML frontmatter + Markdown；显式兼容历史 heading-only 文件 | 只读 | 战役上下文，不是失败真值 |
| `forge/ledger.jsonl` | append-only event stream | Forge 只追加 | 已应用/拒绝决策与 frozen prediction |
| `forge/editable_surface.yaml` | 启动时加载 | 只读治理 | 写面、保护面、阈值与固定验证命令 |
| companion runtime overlay + frozen suite | JSON/YAML + owner validator | 单一候选写面 / suite 只读 | 只允许向既有 `strategy_playbook` owner 添加结构化规则；不拥有语义决策 |
| Common Adapter train/evaluate/gate artifacts | content-addressed JSON | Forge 只读 | 第五阶段只核对构建请求与 `vz-substrate`/cognition 证据，不训练、不发布、不激活 |

契约违反（非法 JSON/YAML、未知 schema version、path 越界、hash 不一致）必须 fail loudly。

## 输出契约

### Failure pattern

正式 schema：`forge/schemas/failure_pattern.schema.json`。

当前输出为 `forge-failure-pattern.v3`，新增 `live_dialogue_outcome` provenance；v1/v2 仍可作为
历史 proposal bundle 输入验证。

每个 pattern 必须包含：

- verifier-level cause；
- agent-behavior cause；
- exposed mechanism；
- evidence refs（source id、locator、有限 excerpt、SHA-256）；
- occurrence count、embedding centroid digest；
- `in-surface` + 唯一 target，或 `out-of-surface` + null target；
- passing behaviors that must be preserved。

cluster 与 target mapping 均使用可注入 semantic embedding；禁止关键词路由或 hash embedding
冒充语义后端。embedding 不可用时命令失败，不静默退化。

### Proposal bundle

目录：`artifacts/forge_propose_<timestamp>/proposals/<proposal_id>/`。

| 文件 | 契约 |
|---|---|
| `patch.diff` | 单目标 unified diff；phase 1 只允许 append-only |
| `manifesto.json` | `proposal_manifesto.schema.json`；证据、根因、修法、预测、风险、保留行为、回滚 |
| `failure_pattern.json` | 生成候选时冻结的 pattern 副本 |
| `validation.json` | `validation_report.schema.json`；每项 PASS/BLOCK 与内容哈希 |
| `gate_decision.json` | 仅 runtime-gated component 使用；`ModificationGate.OFFLINE` 决策、原因与 patch/manifesto/validation 哈希绑定 |

proposal 只表达候选，不拥有目标文件。只有 `apply` 在验证和人审后执行一次正式写入。

### Ledger

`forge/ledger.jsonl` 是 append-only 决策日志。decision event 至少包含 proposal/manifesto/patch
hash、reviewer、decision、timestamp；applied event 还冻结 prediction baseline。拒绝不删除候选，
必须记录原因。ledger 不提供在线 reward，也不回灌 Volvence credit owner。

### Population / rare-heavy artifacts

- `optimizer_decision.json` 遵循 `forge-optimizer-decision.v1`。它按 component 在
  `validation_delta ↑ / capacity_cost ↓ / added_lines ↓ / risk_count ↓` 上计算确定性
  Pareto front；只有 PASS 且所需 OFFLINE gate 为 ALLOW 的候选可入选。空 population 或
  无 eligible candidate 必须发布 `STOP`，不得为了持续循环强选候选。
- `rare_heavy_request.schema.json` 定义 `forge-rare-heavy-request.v1`：绑定 frozen base
  weights digest、traces、control basis、held-out corpus、全部 LoRA/State-KV 超参数和评估阈值；
  `owner=vz-substrate`、`requested_wiring=DISABLED`、`training_decides_gate=false` 为常量。
- `rare_heavy_verdict.schema.json` 定义 loop-external `READY/STOP`。`READY` 只表示请求、
  `common-adapter-candidate.v2`、held-out report 与 cognition OFFLINE ALLOW record 完整绑定；
  它不等于 publish，也不创建 `CommonAdapterBundle`。

### Task-level held-out 诊断

`forge benchmark <target>` 对 `repository_agent_rules` 或 `forge_analysis_prompts` 运行冻结的
`forge-task-benchmark-suite.v1`。基准、判定 prompt、decision/report schema 全部位于 proposal
循环的只读面；每个模型请求只包含当前 harness asset、task 和 structured evidence，不包含
`expected / critical / minimum_confidence` 标签。

报告遵循 `forge-task-benchmark-report.v1`，记录 baseline/candidate asset hash、逐 case 失败、
critical failure 数、pass rate 和 candidate delta。候选必须同时满足绝对 pass-rate 阈值、零 critical
退化和非负 delta，否则 `BLOCK`。该输出固定声明 `diagnostic_only=true`、
`causal_claim_authorized=false`：它用于发现明显决策退化，不进入 `validate/apply` promotion gate，
也不能证明真实开发效率、长期仓库健康或因果收益。

## 优化对象阶梯

开发环继续开放 instruction/structured-context：`.cursor/rules/*.mdc` 与 `forge/prompts/**`。
第三阶段只新增一个精确的产品候选面：
`packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/runtime_assets/companion_playbook_overlay.json`。
它是 additive package-data，不是代码、schema、suite、evaluator 或新 owner；其他 `packages/**`
仍因未进入 allow list 或命中保护面而关闭。第四阶段的 optimizer 与第五阶段的 rare-heavy planner
是人工工程实现，始终位于 Forge 自身提案循环外，不能修改自己、阈值或权限配置。

## 第三至第五阶段收敛结论（2026-08-01）

第二战役预检排除了仅供验收的 character scenario package。第三阶段随后定位到真实产品消费链：
`lifeform-domain-emogpt.build_companion_package()` 构造 `DomainExperiencePackage`，Brain 再把其
`PlaybookRule` 导入 `vz-application` 的 `strategy_playbook` owner。新增 overlay loader 由
`lifeform-domain-emogpt` 拥有，并使用外部 `WiringLevel`：

- `DISABLED`：不读取 overlay，live/candidate 都是 baseline；
- `SHADOW`：读取并严格验证，candidate=baseline+overlay，live 仍为 baseline；
- `ACTIVE`：把 additive rules 编译进既有 package/owner；overlay 不得替换 baseline rule id 或
  problem pattern，也不能在文件内自授权 wiring。

生产 overlay 初始为空且所有 builder 默认 `DISABLED`。Forge 可形成该单文件的结构化
`append_json_array_item /playbook_rules` 候选，但 apply 仍需要 candidate schema/owner validator、
冻结 suite baseline/candidate 对照、component held-in/out、loop-external OFFLINE ALLOW 与 named
human approval。apply 只改变 package-data；从 `DISABLED/SHADOW` 切到 `ACTIVE` 是另一个人工部署
决定，不由候选资产控制。

第四阶段允许每个 failure pattern 生成有界候选 population，并由 `forge select` 发布 Pareto
选择或正式 STOP。selector 重算 patch/manifesto/validation/gate digest，拒绝 stale evidence；
它只选择 proposal id，不 apply。

第五阶段把权重演化限制为“计划请求”：`forge plan-rare-heavy` 只写 content-addressed
DISABLED request。真实训练仍由 `scripts/train_common_adapter_model.py train` 按
`rare-heavy → State-KV → offline-gate` 执行；公开 `validate_common_adapter_evidence()` 复核全部
nested artifact、held-out report 与 cognition gate。`scripts/forge_common_adapter_adjudicator.py`
再核对请求参数与这些证据，只发布 READY/STOP，绝不调用 publish 或修改 runtime wiring。

失败来源采用 typed provenance lane 隔离：

- transcript / promotion verdict 只能映射到 development-harness component；
- `bench_bundle` / `live_dialogue_outcome` 只能映射到显式 `requires_offline_gate` 的 runtime component；
- 两条 provenance lane 不跨 lane 聚类。只有明确映射到
  `companion_runtime_playbook_overlay` 的 runtime observation 才能进入该 component；其他 runtime
  failure 仍为 `out-of-surface`，不得退回开发 rules/prompts。

live outcome 不是自动 failure label。`mine --live-outcome-root <dir>` 只验证并提交 typed metadata；
结构化语义 backend 可以返回零条 failure record，且不得重建 artifact 已刻意删除的对话内容。
`--evidence-since` 对该来源使用 artifact 内 content-bound `recorded_at_iso`，不信任可变文件 mtime。

runtime gate 的预注册折算保持冻结：

- `validation_delta = candidate_pass_rate - baseline_pass_rate`，OFFLINE 最低增益为 `0.05`；
- 文件级候选 `capacity_cost = 0.1`；
- `contract_integrity = 1.0` 仅当 target 未触及冻结 suite/阈值且契约检查通过；
- `rollback_resilience = 1.0` 仅当 reverse patch 后与原文件逐字节相同；
- apply 必须同时满足同一 bundle 的 gate `ALLOW`、输入哈希一致和 named human approval；任一缺失
  或不一致都 fail closed。

`scripts/forge_gate_adjudicator.py` 位于循环外并调用 cognition owner 的 OFFLINE gate；Forge
本体继续禁止 import `vz-*`。该裁决器只为精确 companion overlay 提案激活；第五阶段 Common
Adapter 使用独立的 `forge_common_adapter_adjudicator.py`，两种 gate artifact 不可互换。

## 三类可观测性

1. component：可编辑资产在文件系统有唯一 path、component id 与语义描述；
2. experience：raw source → failure record → failure pattern 分层，支持由 pattern 回溯有限证据；
3. decision：每个 edit 同时冻结预测与 at-risk regression，下一轮 mine 发布兑现检查。

## 关键不变量

1. 保护面检查优先于白名单；allow 不能覆盖 deny。
2. 除精确 companion overlay JSON 外，packages、tests、gate/verifier scripts、spec、
   DATA_CONTRACT、artifacts、Forge code/schema/config/ledger 永远不在当前循环可编辑面。
3. Forge 顶层不得 import `volvence_zero.*` 或 `lifeform_*`；业务 wheel 不得 import
   `volvence_forge`。
4. evaluator、permission control、LLM configuration 与 validation commands 位于循环外。
5. proposal 不自动 apply；PASS validation + named human reviewer 缺一不可。
6. phase 1 diff 单文件、append-only；preimage、patch、manifesto 在 propose/validate/apply 间做
   SHA-256 复核。
7. 任一验证缺失、超时、judge 不确定、schema/path/hash 不匹配都 fail-closed BLOCK。
8. prediction check 只能报告观测计数是否满足 frozen expectation，不把相关性宣称为因果成功。
9. 失败与拒绝是正式证据；不得为提高成功率删除 ledger 负结果。
10. prediction evidence 未显式晚于 applied event 时必须为 `inconclusive`，不得让历史样本污染
    `fulfilled/refuted`。
11. bench evidence 没有 OFFLINE-gated runtime owner 时必须 `out-of-surface`；禁止退而映射到开发
    rules/prompts。
12. optimizer 无 eligible candidate 必须 STOP；selector 不能 apply，rare-heavy request/verdict
    不能 publish 或激活 bundle。
13. live outcome 目录只能显式提供；typed outcome 不等于 failure，parser 不保存或重建原始对话，
    content hash、隐私 profile 或 recorded timestamp 不合规时必须 fail loudly。
14. task benchmark 的 suite/prompt/schema 对提案循环只读，case label 不进入模型输入；synthetic
    PASS 只能形成诊断证据，不能授权 apply 或 runtime promotion。

## 验证契约

`validate` 顺序：

1. schema 与 bundle 一致性；
2. diff path 单一且落在白名单，未命中保护面；
3. target preimage hash 一致；
4. `git apply --check`；
5. `.mdc` 无删除行（phase 1 hard-constraint preservation）；
6. 循环外 relevance judge 对 failure pattern/manifesto/diff 给出全部肯定；
7. runtime component 的候选先在临时文件执行 owner/schema validator，再运行冻结 suite 对照；
8. component-specific held-in/out 与 frozen static commands；
9. held-in Forge tests；
10. held-out boundary contract tests。

检查不短路：报告尽可能记录全部失败，但任一 BLOCK 使总状态 BLOCK。

## 时间尺度与 Volvence 关系

Forge 是 out-of-turn development/background 工具，不进入 `online-fast`、`session-medium` 或
`background-slow` runtime。开发环编辑相当于人工审核的 rare process，但不是 substrate
rare-heavy artifact；产品 runtime 的自修改仍必须走 cognition owner 的 `ModificationGate`。

Evaluation 在此只提供只读 gate evidence（R12），不会反向成为 PE 或 runtime credit 源。

## 迁移、退出与回滚

phase 1 完成条件：

- 真实 93 transcript + 至少一份失败 promotion verdict 可完成 mine；
- 至少一个 in-surface pattern 可形成完整 proposal bundle；
- validate 对合规 fixture PASS、对越权/删除/篡改 fixture BLOCK；
- `ruff check forge/ tests/contracts/test_forge_boundaries.py` 通过；
- `pytest forge/tests tests/contracts/test_forge_boundaries.py` 通过。

回滚 Forge：删除 `forge/`、本 spec、索引项与 boundary test。回滚已应用候选：验证 preimage
关系后在仓库根目录执行 bundle 记录的 repo-relative reverse patch 命令；若已提交，创建独立
revert commit。ledger 保留审计史。

companion overlay 的第三阶段进入条件已由真实 DomainExperiencePackage consumer、外部 wiring、
owner validator、frozen suite、rollback drill 和 OFFLINE gate 满足；因为它编译到既有
`strategy_playbook`，没有新增跨模块交换，故不注册新 DATA_CONTRACT slot。退出时将 wiring 设为
`DISABLED` 即恢复 baseline；若已 apply 某 rule，再按 proposal reverse patch 或独立 revert
恢复空/前一 overlay。扩展第二个 runtime asset 仍须重复同样 owner/consumer/slot 审查，不能复用
本例授权。

## 已知限制

- 已建立冻结 synthetic task-decision held-out split，可对 harness baseline/candidate 做 exact scoring；
  尚未建立执行真实仓库任务、由独立 verifier 裁决的 causal benchmark，因此 diagnostic PASS 仍不等于
  真实开发效率或长期成功率已提升。
- transcript 的结构化错误并不总能支持强因果分析；低置信记录必须保留不确定性。
- proposal 语义 judge 仍是模糊 evaluator，因此 judge 只能收紧，不可单独授权 apply。
- Pareto 目前使用四个冻结工程指标，不声称已经学习到 optimizer；也不会自动繁殖下一代。
- companion overlay 的能力链已开放，产品服务现可通过显式 CLI 进入 SHADOW 并在监听前验证候选；
  生产资产仍为空、默认 DISABLED，服务边界拒绝 ACTIVE，本阶段没有执行 runtime apply 或 ACTIVE
  部署。
- rare-heavy 请求/裁决契约已落地，但本阶段没有可用的冻结模型 snapshot、训练 trace、GPU run 与
  新 held-out ALLOW 证据，因此没有生成或发布新的 CommonAdapterBundle。
- live mine/propose 仍依赖显式 `FORGE_LLM_API_KEY`/`FORGE_LLM_MODEL`；无凭据时只运行 replay
  契约演练，不把它宣称为真实模型晋级证据。产品 typed outcome source 已打通，但当前工作区尚无
  实际 opt-in closed-alpha outcome artifact，不能据此宣称 live prediction check 已完成。

## 参考

- Lilian Weng, Harness Engineering for Self-Improvement（本地只读归档：
  `docs/external/lilian-weng-harness-engineering-2026-07-04.html`）
- `docs/specs/credit-and-self-modification.md`
- `docs/next_gen_emogpt.md` R8/R10/R12/R15
- `archetecture.md`
