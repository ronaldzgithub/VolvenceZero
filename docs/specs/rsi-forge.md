# RSI Forge 元层 Spec

> Status: package 0 skeleton and boundary contract
> Last updated: 2026-08-01

## 职责与 owner

`forge/` 是仓库级离线元层，唯一职责是把开发环中的失败证据压缩为可审计的、
白名单约束的编辑提案。它不是 runtime module，不拥有 prediction error、evaluation、
memory、semantic state、ModificationGate 或任何产品状态；这些数据只能作为只读
输入消费。Forge 的 proposal ledger 是元层 provenance，不是 Volvence 的学习 ledger。

第一阶段编辑面只包含 `.cursor/rules/*.mdc` 与 `forge/prompts/**`。`packages/**`、
`tests/**`、gate/evaluation/verifier 脚本、`docs/specs/**`、`forge/src/**`、schema、
白名单、配置与 ledger 都是只读面。扩展编辑面必须由人工修改
`forge/editable_surface.yaml`，不能由 Forge 自己提出或应用。

## 输入、输出与阶段边界

输入是已存在的结构化 agent transcript、promotion/evaluation artifact 和
`.cursor/plans/*.plan.md`。mine 阶段输出 `failure_pattern`，至少分离 verifier
层原因、agent 行为因果和暴露的 harness 机制；propose 阶段输出 unified diff 与
proposal manifesto；validate 阶段只做 fail-closed 静态/回归检查；apply 阶段必须
经过人工确认后才写入白名单文件，并把决策与预测追加到 `forge/ledger.jsonl`。

这些阶段都是 background-slow/offline 工作，不阻塞实时 turn，不自动改变生产
WiringLevel。第二阶段若要接产品 runtime，必须另开收敛包，注册 DATA_CONTRACT slot，
并经过 `ModificationGate.OFFLINE`、SHADOW 对照、独立 evidence 和可回滚部署。

## 共享 schema

- `failure_pattern`：稳定 id、来源引用、三层因果记录、owner/编辑面映射、置信度、
  不确定性和去重指纹；无法映射到白名单时必须标记 `out-of-surface`，只能报告。
- `proposal_manifesto`：目标文件、pattern 引用、根因、窄修复、预测影响、风险回归、
  验证命令、回滚方式和人工决策；不能缺少证据引用。
- `validation_report`：白名单结果、patch 可应用性、held-in/held-out 命令、失败原因
  和回滚摘要；任何未知字段、命令失败或超时均为 BLOCK。
- `ledger`：append-only 事件流；proposal 的 apply/reject/revert、预测兑现和审阅
  者均可追溯，不允许被候选提案覆盖。

## 不变量与退出条件

1. AST 边界测试必须保证 `packages/**` 不 import `volvence_forge`，而 `forge/src/**`
   不 import `volvence_zero.*` 或 `lifeform_*`。
2. 任何提案目标不在白名单、触碰只读面、绕过验证器、删除硬约束或缺失 manifesto
   字段都直接 BLOCK；禁止静默回退。
3. 关系质量、human rating 和 evaluation readout 不得写入 reward、credit 或 Forge
   的编辑决策源；它们只能作为独立证据或校准输入。
4. 第一阶段完成条件是 boundary/forge 自测通过，且能对真实失败 artifact 产出一份
   proposal bundle；没有人工确认不得 apply。若语义后端不可用，mine 只能报告明确
   的 unavailable 状态，不得使用关键词路由替代。
5. 回滚是删除未应用的 Forge artifact，或对已应用 patch 执行 Git revert；ledger
   保留完整历史。Forge 本身不授权任何 production/live promotion。

## 验证

包 0 使用 `ruff check forge tests/contracts/test_forge_boundaries.py` 与
`pytest tests/contracts/test_forge_boundaries.py`。后续每个包只运行其直接相关测试；
涉及公共契约或 import boundary 时追加 `pytest tests/contracts`。
