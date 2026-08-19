# relationship_transfer_v1

Relationship Lab v0 的内容寻址场景包。它只用于离线证据，不进入 companion
产品路径。

数据严格分层：

- `rendered_observations.json` 是唯一允许构造 SUT 输入的数据；
- `generator_truth.json` 只允许反应式环境与只读评估器访问；
- `RelationshipDecisionTrace` 在动作和 outcome 都发生后生成，记录行动前下注、
  实际动作、环境证据与 lineage。

`manifest.yaml`、`ssot_fragment.json`、`scenes.yaml` 与 `test_suite.yaml`
固定路径、R14 体制身份和语义路由纪律。`prereg_template.json` v4 固定 P0/P1b 后续
formal 必须封存的 substrate、system prompt、request template、readout schema、
typed compiler、RAG、seed、阈值与 arm 对照；模板本身不是一份已冻结 prereg。

包内标为 `heldout` 的 pair 只是开发期结构分割：它验证 loader 隔离，并禁止
stateless calibration 消费对应 pair。由于 `generator_truth.json` 位于仓库中，它
不是 formal secret heldout；正式盲测集必须在 prereg 冻结后另行生成和封存。

本包刻意不提供从文本到动作的关键词规则，也不把 hidden dynamic 作为模型
输入。默认 Gate 0 校准只能得到 `machinery_ready=true`；只有真实冻结 substrate
的 stateless/raw baseline attestation 满足样本量、100% structured-output validity
和预注册 non-saturation ceiling，Gate 0 才能通过。
