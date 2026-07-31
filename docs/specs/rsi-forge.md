# RSI Forge Spec

> Status: draft / development-loop phase 1
> Last updated: 2026-08-01
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
| `.cursor/plans/*.plan.md` | YAML frontmatter + Markdown；显式兼容历史 heading-only 文件 | 只读 | 战役上下文，不是失败真值 |
| `forge/ledger.jsonl` | append-only event stream | Forge 只追加 | 已应用/拒绝决策与 frozen prediction |
| `forge/editable_surface.yaml` | 启动时加载 | 只读治理 | 写面、保护面、阈值与固定验证命令 |

契约违反（非法 JSON/YAML、未知 schema version、path 越界、hash 不一致）必须 fail loudly。

## 输出契约

### Failure pattern

正式 schema：`forge/schemas/failure_pattern.schema.json`。

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

proposal 只表达候选，不拥有目标文件。只有 `apply` 在验证和人审后执行一次正式写入。

### Ledger

`forge/ledger.jsonl` 是 append-only 决策日志。decision event 至少包含 proposal/manifesto/patch
hash、reviewer、decision、timestamp；applied event 还冻结 prediction baseline。拒绝不删除候选，
必须记录原因。ledger 不提供在线 reward，也不回灌 Volvence credit owner。

## 优化对象阶梯

phase 1 只开放 instruction/structured-context：`.cursor/rules/*.mdc` 与 `forge/prompts/**`。
workflow、Forge code、optimizer code、产品 runtime 与权重是后续独立 convergence packet；任何
扩面必须由人类先修改治理文件、补 verifier 与回滚证据，不能由本循环提案。

## 三类可观测性

1. component：可编辑资产在文件系统有唯一 path、component id 与语义描述；
2. experience：raw source → failure record → failure pattern 分层，支持由 pattern 回溯有限证据；
3. decision：每个 edit 同时冻结预测与 at-risk regression，下一轮 mine 发布兑现检查。

## 关键不变量

1. 保护面检查优先于白名单；allow 不能覆盖 deny。
2. packages、tests、gate/verifier scripts、spec、DATA_CONTRACT、artifacts、Forge code/schema/config/
   ledger 永远不在当前循环可编辑面。
3. Forge 顶层不得 import `volvence_zero.*` 或 `lifeform_*`；业务 wheel 不得 import
   `volvence_forge`。
4. evaluator、permission control、LLM configuration 与 validation commands 位于循环外。
5. proposal 不自动 apply；PASS validation + named human reviewer 缺一不可。
6. phase 1 diff 单文件、append-only；preimage、patch、manifesto 在 propose/validate/apply 间做
   SHA-256 复核。
7. 任一验证缺失、超时、judge 不确定、schema/path/hash 不匹配都 fail-closed BLOCK。
8. prediction check 只能报告观测计数是否满足 frozen expectation，不把相关性宣称为因果成功。
9. 失败与拒绝是正式证据；不得为提高成功率删除 ledger 负结果。

## 验证契约

`validate` 顺序：

1. schema 与 bundle 一致性；
2. diff path 单一且落在白名单，未命中保护面；
3. target preimage hash 一致；
4. `git apply --check`；
5. `.mdc` 无删除行（phase 1 hard-constraint preservation）；
6. 循环外 relevance judge 对 failure pattern/manifesto/diff 给出全部肯定；
7. frozen static commands；
8. held-in Forge tests；
9. held-out boundary contract tests。

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

扩展到 runtime 的进入条件：独立 spec、DATA_CONTRACT slot/owner（若新增交换）、
`ModificationGate.OFFLINE`、SHADOW 对照、rollback drill 和任务级 held-out evidence 全部先行。

## 已知限制

- phase 1 的 task-level held-out benchmark 尚未建立；结构 PASS 不等于真实开发效率已提升。
- transcript 的结构化错误并不总能支持强因果分析；低置信记录必须保留不确定性。
- proposal 语义 judge 仍是模糊 evaluator，因此 judge 只能收紧，不可单独授权 apply。
- 多候选搜索只做 embedding 去重，尚未实现 Pareto population 或 STOP 式 optimizer evolution。

## 参考

- Lilian Weng, Harness Engineering for Self-Improvement（本地只读归档：
  `docs/external/lilian-weng-harness-engineering-2026-07-04.html`）
- `docs/specs/credit-and-self-modification.md`
- `docs/next_gen_emogpt.md` R8/R10/R12/R15
- `archetecture.md`
