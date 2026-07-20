# 统一合成经验母语料 Spec

> Status: stable-v1
> Last updated: 2026-07-20
> 对应需求: R1–R8、R11–R15、R16–R20、R-PE

## 要解决的问题

为关系编码、表达 SFT、语义/社会 owner、连续记忆、ETA/内部 RL 和只读评估提供同一份可追踪母语料，避免各训练任务各自重建真值、混淆模型输出与世界事实，或把 Companion Bench held-out 评测数据引入训练。

该能力属于 proprietary 离线数据工件，由 `lifeform-synthetic-data` wheel 唯一拥有。它不是 runtime owner，不发布新 slot，不改变 `docs/DATA_CONTRACT.md` 的运行时依赖图；运行时只通过 `Lifeform` / `BrainSession` 公共 facade 导出快照读数。

## 关键不变量

1. 母记录以 frozen dataclass 为 SSOT；canonical JSON、JSON Schema 与稳定 SHA-256 必须由同一结构导出。
2. 三层证据物理分离：
   - `generator_truth`：FSM/world compiler 已知的可观察事实、私有状态、目标、边界和转移，可作为硬标签；
   - `rendered_text`：LLM 只能回填已有稳定 ID 的文本槽，不得修改状态或标签；
   - `runtime_observation`：公开不可变快照的原始序列与 hash，只是运行时观测，不冒充世界真值。
3. 每条 annotation 必须声明 ontology/version、target、owner/track/timescale/scope、source、confidence、evidence、adjudication 与 training use。
4. `MODEL_PREDICTION` / `EVALUATION_READOUT` 默认只能 `feature_only`、`eval_only` 或 `quarantined`，不得成为训练目标；没有真实人工流程时不得写 `HUMAN_ANNOTATION`。
5. `z_t` / `beta_t` 只按 temporal owner 快照原样记录，不手工赋予语义标签；内部 RL 视图只来自 live-through 的 segment/PE/credit 链。
6. 场景选择使用显式 scenario id 或结构化语义分类输出，不从自然语言关键词路由。
7. 每个派生视图携带 master trajectory hash 与字段级 source refs，不维护第二份真值。
8. held-out benchmark、真实 PII、秘密、无授权版权语料和 API key 永不进入母语料。
9. 任一硬质量门禁失败即停止扩容；失败样本写 quarantine/ledger，不以删除失败样本缩小分母。

## 母 schema

`lifeform_synthetic_data.contracts` 定义以下不可变类型：

- `CorpusManifest`
- `ScenarioBlueprint`
- `ExperienceTrajectory`
- `ExperienceSession`
- `ExperienceTurn`
- `LatentTruthFrame`
- `SnapshotFrame`
- `AnnotationRecord`
- `ArtifactRef`
- `QualityRecord`

`ExperienceTrajectory.schema_version` 固定为 `synthetic-experience.v1`。严格反序列化必须拒绝未知字段、缺失字段、非法 enum、坏引用和重复 ID。canonical JSON 使用 UTF-8、排序 key、无无意义空格与 `allow_nan=False`；artifact、scenario、prompt、snapshot、shard 均以 SHA-256 寻址。

## 场景包

内置 `unified_v1` 包含 `manifest.yaml`、`ssot_fragment.json`、`scenes.yaml`、`test_suite.yaml`。16 个能力族各有 4 train、1 val、1 test，共 96 个 blueprint。同一 persona、latent arc、翻译/改写和 counterfactual sibling 不跨 split。

该包与现有 `vz-scenario-pack.v1`、Companion Bench `ScenarioSpec` 是不同契约；转换必须通过显式 adapter，禁止 duck typing 或把两套格式当同一对象。

## 生成接口

生成分三层：

1. `structural`：确定性 world/FSM compiler 生成事件、状态转移、硬标签与结构占位文本；
2. `rendered`：集中式 prompt + JSON Schema 调用 OpenAI-compatible 模型，仅替换文本槽；
3. `live_through`：只从已完成并通过门禁的 rendered master 做按 scenario 分层的确定性抽样，逐 turn 调用 `LifeformSession` 公共接口并采集公开快照，生成 runtime observation sidecar；必须保留 source master trajectory hash/run ID，禁止重新编译结构占位文本冒充真实输入。

批处理必须支持 seed、shard、append-only journal、幂等 resume、失败 quarantine、调用/token/费用记账、用户提供 rate card 和 `--max-cost-usd` 硬上限。鉴权、配额、合同或 schema 错误 fail loudly；只允许对文档化的瞬时 HTTP 状态重试。

## 派生视图

- `relationship_encoder`：投影到 `companion_standard.InteractionTrajectory`，标签只来自 generator truth；
- `expression_sft`：只保留通过 response contract 硬门禁的 teacher 文本；
- `semantic_owner` / `social_cognition`：从 typed annotation 投影 proposal/状态目标；
- `memory_retrieval`：由事实时间线与 subject/audience scope 生成 query/positive/hard-negative；
- `temporal_ssl` / `internal_rl`：只消费 live-through snapshot/segment/PE/credit；
- `evaluation_only` / `human_review_queue`：物理隔离于训练视图。

每个视图有独立 split manifest。关系编码器仍采用更严格的 scenario-family 隔离。

## 质量门禁

最少覆盖：schema/immutability/hash、96 场景与引用完整性、phase 顺序、split lineage 零泄漏、annotation source policy、judge 不入训练、held-out/版权来源隔离、PII/secret、精确与近重复、分布、幂等 resume、费用门禁、HTTP 故障和端到端 projection。

扩容顺序固定为 96 structural golden → 96 rendered pilot → 768 扩容验证 → 10,240 rendered master → 1,024 分层 live-through。每一阶段产出 run manifest、checksum、dataset card、字段字典、annotation handbook、coverage matrix、费用/模型分布和审计报告。

## 与其他能力域的关系

- `companion-standard` 只提供公开关系 schema；本 wheel 不扩宽其职责。
- `companion-trajgen` 保持 OSS 合成关系轨迹生成器，不包含 proprietary 场景或生成工艺。
- `lifeform-evolution` 只消费投影视图，不拥有母语料真值。
- runtime owner 保持原有 SSOT；本 wheel 只能读取 facade 导出的不可变快照。
- evaluation/judge 是只读审计，不回灌标签、PE、credit 或控制器更新。

## 变更日志

- 2026-07-20：冻结 v1 母 schema、96 场景包、三层生成路径、投影与质量门禁。
