# Live Dialogue Outcome Evidence

> Status: closed-alpha opt-in ACTIVE；Forge read-only ingestion 已接入

## 职责

`lifeform-service` 把已经由 runtime owner 接受的 `DialogueExternalOutcomeEvidence` 导出为
去标识化、不可覆盖的产品反馈 artifact，为后续 Forge failure mining 提供真实产品来源。
该出口不解释用户文本、不判断 failure、不写 memory / PE / credit，也不创建新的 runtime owner。

## 启用与数据边界

只有 `AlphaServiceConfig.enabled=True` 且显式配置 `evidence_root_dir` 时，公开
`POST /v1/sessions/{id}/dialogue-outcomes` 才写 artifact。普通产品服务、未配置 evidence root 的
closed alpha、以及其他内部 outcome producer 均不落该文件。回滚时去掉 evidence root 或关闭
alpha 即停止新增 artifact，不影响 outcome 进入 runtime 的既有路径。

artifact schema 为 `lifeform-live-dialogue-outcome.v1`，输出到
`<evidence_root>/live_dialogue_outcomes/<sha-prefix>/<source-evidence-sha>.json`。它只包含：

- owner-issued outcome kind/source/confidence、consuming turn 与 action turn；
- action turn 的 compact `TurnSummary` readout：regime、abstract action、PE magnitude、open-loop /
  commitment 数量、trigger kind 与 tick；
- service/policy version、UTC 记录时间、identity/session/scene/evidence 的 SHA-256；
- 对整个 payload 的 `content_sha256`。

明确禁止保存 `user_input`、`response_text`、自由文本 description、原始 evidence_ref、明文 user /
tenant/session/scene id。文件使用 create-only 写入；同一 evidence 重试必须验证既有 content hash，
篡改或稳定字段冲突时 fail loudly。

## Owner 与消费边界

- `DialogueExternalOutcomeEvidence` 的语义 owner 仍是 runtime 的
  `dialogue_external_outcome`；service 只投影公共不可变 value。
- `TurnSummary` 仍由 `lifeform-core` 发布；service 不遍历 kernel 内部状态，也不重建 owner。
- artifact 是 out-of-turn export，不是模块间 snapshot，所以不新增 `docs/DATA_CONTRACT.md` slot。
- Forge 通过显式 `--live-outcome-root` 把它当只读 observation source；是否属于失败必须走 LLM 结构化语义挖掘或明确的
  typed evaluator，不得在 service/Forge 中加入关键词路由。
- Forge 不自动发现该隐私目录；parser 复核 exact fields、content hash、identity hashes、UTC recorded
  time 与 action context。语义 backend 可以返回零条 failure record，因此 typed outcome 不会被代码
  自动定罪。接入 parser 仍不等于 live remine 已完成；后者还需要真实 artifact 和 LLM 凭据。

## 验证与回滚

- 覆盖去标识化、action-turn 归因、content hash、幂等 create-only 与篡改阻断。
- handler 测试证明只有显式 closed-alpha evidence 配置返回 artifact ref。
- 关闭 alpha/evidence root 即退出；删除本 sink 与 handler 调用可代码回滚，既有 runtime outcome
  行为不变。历史 artifact 继续受 evidence deletion/retention policy 管理。
