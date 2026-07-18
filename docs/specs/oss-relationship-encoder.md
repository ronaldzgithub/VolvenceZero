# OSS Option B — Open-Weights Relationship Encoder Spec

> Status: draft（依赖 Option A 先行；训练为新研究项目，非现有代码直接发布）
> Last updated: 2026-07-18
> 对应需求: R2（稳定基底 + 自适应控制器）、R8（快照优先）、R11（内部状态可发布）、
> R12（评估只读）、R15（可回滚 + 证据先行）
> 前置依赖: `docs/specs/oss-relationship-representation-standard.md`（Option A）——
> encoder 的输入 / 输出 schema 与 conformance 全部锚定该标准，**标准不公开则本项目不启动**。
> 认知基线: `docs/relationshipdynamics.md`——通用动力学 vs 个体轨迹的分层论证与三条不变量
> （权重只装跨人共享规律 / 开源切口 = R2 切口 / 通用性主张须过 G2 才可对外使用）。
> 商业动机: 「开源人类表征大模型」叙事中"模型"一词的诚实兑现——有权重、有评测、有数据管线，
> 才配得上 model 这个词；同时为标准提供第一个非平凡的开源实现，反衬商业运行时的差距。

## 要解决的问题

训练并开源一个小型**关系编码器**：输入一段（多 session 的）人机交互轨迹，输出符合
Relationship Representation Standard 的关系状态表示——关系阶段、信任走向、未闭环承诺、
边界事件、rupture/repair 状态的结构化预测 + 一个 trajectory-level 向量表征。

**与 DINQ 的对象区分**：DINQ 的 person embedding 表征"社会中的这个人是谁"（静态社会图谱）；
本 encoder 表征"一段关系走到了哪里"（动态交互轨迹）。不做人物画像，不做身份识别。

**与内核的对象区分**：本 encoder 是**离线的、无状态的 readout 模型**——给轨迹打关系状态标注。
它不是运行时 owner，不持有记忆，不做在线适应。内核的 semantic owner / metacontroller /
CMS 与它零共享实现。

## 模型契约

| 项 | 契约 |
|---|---|
| 输入 | canonical trajectory JSON（Option A 标准定义的多 session 交互轨迹格式） |
| 输出 1 | 结构化预测：符合标准 schema 的 `RelationshipStateSnapshot`-兼容字段子集（关系阶段 / 信任方向 / rupture 状态 / 未闭环承诺计数） |
| 输出 2 | trajectory embedding（定长向量，维度 v0 定档后冻结） |
| 接入接缝 | 实现标准 wheel 公开的 `SemanticEmbeddingBackend` Protocol（作为其一个 backend） |
| 规模 | 小模型优先（≤1B，基于开源基底 fine-tune 或从头训小 encoder，v0 选型见 §Open Decisions） |
| 发布物 | 权重（HF Hub）+ 训练/评测代码 + **合成数据生成管线**（不发布数据集本体的真实来源材料，见数据边界） |

## 关键不变量

1. **数据边界（最高优先级，违反即撤版）**：
   - 训练数据 **100% 合成或已获授权**。任何 per-tenant / closed-alpha / JV 场景的真实用户
     数据禁止进入训练集、评测集、示例、README。
   - 发布的是**数据生成管线**（persona 模板 + FSM + LLM simulator 配置），第三方可自造数据；
     我们自用的生成产物可发布抽样，需经清洗 checklist。
   - held-out（`external/companionbench-heldout/`）永不进入训练集——它是评测的私有防污染层。
2. **评估只读（R12）**：Companion Bench 是 encoder 的评测面，不是训练信号源。
   禁止在 held-out 或公开场景的 judge 输出上直接训练（防止把 benchmark 蒸馏进权重）。
   训练标签来自数据生成管线的 ground-truth FSM 状态（生成时天然带标签），不来自 judge。
3. **不泄漏内核工艺**：encoder 训练代码不 import `volvence_zero.*` / `lifeform_*`
   （守门测试强制，同 companion-bench 模式）。metacontroller / internal RL / CMS /
   owner 实现的任何代码、权重、蒸馏产物不进入发布物。
4. **不成为第二 owner（R8）**：商业运行时**不**将 encoder 输出直写任何 semantic owner。
   若运行时要消费它，走标准的外部证据入口（typed proposal → owner 审核），
   与 DINQ People Graph 接入同一边界（`dinq-bp-inspiration-2026-07.md` §4）。
5. **证据先行（R15）**：发布 gate 见 §Release Gates；任何 gate 未过，权重不出仓。

## 训练数据管线（全部基于现有代码扩建）

```text
persona / FSM 场景生成（扩建 companion-bench spec.py + scenarios 模式）
  → LLM user simulator × SUT 对话推演（复用 user_simulator.py + arc_runner.py）
  → 生成时记录 FSM ground-truth 关系状态序列（establish/rupture/repair/…16 动作词表已有）
  → character live-through 管线补充长篇叙事轨迹（扩建 lifeform-domain-character 的
     chapter_replay / chapter_experience，输入用公版文本，替换现有受版权语料）
  → canonical trajectory JSON（Option A schema）+ 逐段关系状态标签
  → train / val 划分（held-out 场景族整族隔离，防结构泄漏）
```

规模目标 v0：10^4–10^5 条轨迹（合成可再生，规模由生成预算决定，先小后大）。

**版权注意**：`data/novels/` 现有语料（如金庸文本）**不得**进入开源管线；
live-through 开源分支只用公版 / 自造叙事材料。

## 评测协议

| 评测 | 内容 | 对照 |
|---|---|---|
| 结构化预测准确率 | 关系阶段 / rupture / 承诺状态 vs 合成 ground-truth | 多数类 baseline + GPT 级 LLM zero-shot 打标 |
| Companion Bench readout | encoder 预测与 arc judge A1-A6 的相关性（只读，不训练） | `companion-ref-harness` KV baseline |
| Embedding 质量 | 轨迹检索：同 persona 不同阶段 / 不同 persona 相似阶段的区分度 | 通用 sentence embedding（如开源 text encoder）直接编码轨迹 |
| 校准 | 预测置信度校准曲线（对外引用任何"准确率"数字的前置条件） | — |

对外引用规则沿用四分口径（定性 / 模型分数 / 校准概率 / 已验证结果）：
**模型卡上只出现带校准与对照的数字**。

## Release Gates（顺序不可跳）

| Gate | 条件 | 未过的动作 |
|---|---|---|
| G1 数据审计 | 训练集溯源审计：100% 合成/授权，版权与 PII 扫描通过 | 阻塞一切后续 |
| G2 效果下限 | 结构化预测显著优于 LLM zero-shot 打标对照（否则"模型"名不副实） | 回炉或降级为 Option A 的 conformance 示例 |
| G3 泄漏审计 | 权重与代码不含内核工艺蒸馏路径；守门测试 + 人工 review | 阻塞发布 |
| G4 口径审计 | 模型卡 anti-claims 过一遍（见下） | 阻塞发布 |

## 对外口径纪律（anti-claims）

- 名称用 **relationship encoder / relationship representation model**；
  不用"人类表征大模型"直译（我们不表征"人"，表征"关系轨迹"；且 ≤1B 不称"大"模型）。
- 不称"理解了人"；模型卡明示：在**合成分布**上训练，真实分布泛化未验证即如实标注。
- 不暗示它是 Volvence 商业运行时的内核组件——它是标准的开源参考 backend，
  商业运行时的关系智能来自另一套（闭源的）在线架构。
- "human world model" 措辞继续受 evidence gate 约束，本项目不解锁该词。

## 工程分解（Milestone，供排期用）

| M | 交付 | 依赖 |
|---|---|---|
| M0 | Option A Phase A1 合入（schema SSOT 就位） | — |
| M1 | 数据管线 v0：companion-bench 生成器扩建 + 带标签轨迹 10^4 条 | M0 |
| M2 | encoder v0 训练 + 内部评测（G2 预检） | M1 |
| M3 | live-through 公版语料分支 + 数据规模扩到 10^5 | M1（与 M2 并行） |
| M4 | 四项评测全跑通 + 校准报告 | M2 |
| M5 | G1-G4 逐 gate 过审 → HF Hub + 公开 repo 发布 | M4 + Option A Phase A2 |

预估周期 3-6 个月（M1-M2 是关键路径；若 G2 不过，止损点在 M2，损失限于数据管线——
而数据管线本身可降级为 Option A conformance 测试数据生成器，沉没成本有限）。

## 接口契约

- **消费输入**：Option A canonical trajectory JSON；合成数据管线的 ground-truth 标签。
- **产出输出**：开源权重、`SemanticEmbeddingBackend` 实现、结构化关系状态预测、模型卡。
- **明确不产出**：运行时状态写入、在线学习信号、任何进入内核 owner 的直写路径。

## 与其他能力域的关系

- **Option A（前置）**：schema、conformance、embedding 接缝全部来自标准 wheel。
- **Companion Bench**：只读评测面 + 数据生成模式来源；held-out 永不入训练。
- **4 连续记忆 / 6A semantic owner / 3 时间抽象**：零实现共享；encoder 不是这些能力域的
  替代或第二实现。
- **vz-substrate text_encoder**：商业运行时的私有 backend 与本开源 backend 是同一接缝的
  两个实现，互不依赖。

## Open Decisions（启动前必须拍板）

| # | 决策 | 建议 |
|---|---|---|
| 1 | 基底选型：开源 LM fine-tune（Qwen 系小模型）vs 从头训小 encoder | fine-tune 起步，G2 快速验证；从头训留给 v1 |
| 2 | 输出 2（embedding）v0 是否必须，还是先只发结构化预测 | 都发；embedding 是"表征模型"叙事的核心，砍掉则退化为分类器 |
| 3 | 数据生成的 LLM simulator 预算与供应商（成本主项） | 与 Companion Bench 现有 simulator 配置共用采购口径 |
| 4 | 是否接受社区训练数据贡献（conformance 过了就收？） | v0 不收，G1 审计面失控；v1 再议 |
| 5 | 发布主体与品牌（Volvence 官方 vs 独立标准组织形象） | 与 Option A Open Decision #1（命名族）一起定 |

## 实施记录（2026-07-18，M0 + M1-v0 完成）

M0（标准 SSOT 就位）与 M1 数据管线 v0 已落地为 `packages/companion-trajgen/`
（模块 `companion_trajgen`，Apache-2.0，依赖 `companion-standard` + `companion-bench`；
独立包而非并入 companion-bench，保持 bench 评测纯度）。与草稿的对齐/差异：

- **FSM ground-truth 标注器**（`labeler.py`）：16 动作词表 → 确定性关系状态游走
  `(phase, trust, continuity, repair_pressure)`，逐 session 起点 + 逐 FSM anchor 发标签；
  未登记动作 fail-loud（词表增长是 RFC-level 变更）。标签来自生成时 FSM 状态，
  judge 不可入标签（`LabelSource` 类型级排除 + AST 守门
  `test_companion_trajgen_only_fsm_ground_truth_label_source`）。
- **轨迹导出**（`exporter.py`）：`ArcRecord` → canonical `InteractionTrajectory`，
  bench 1-based (session, turn-pair) 坐标 → 标准 0-based 平铺 turn 坐标
  （user turn t → trajectory index (t-1)*2）；每条导出前过 conformance round-trip。
- **双模式**：`fsm`（DeterministicFakeUtteranceClient + 包内 DeterministicFakeSUTClient，
  零 LLM 成本，两次运行字节级一致已验证）；`llm`（bench 的 OpenAIUtteranceClient +
  OpenAIChatClient，采购口径与 bench 共用）。CLI：`companion-trajgen generate`。
- **held-out 结构性排除**：禁止 import `heldout_loader` + 所有 `load_scenarios_dir`
  调用 AST 强制 `include_held_out=False`（`test_companion_trajgen_boundaries.py`）。
- **M1 验收**：公开 30 场景 × 3 seeds = 90 条轨迹全部通过 `check_trajectory_document`；
  train/val 按场景族整族划分（默认 val = F5, F6），无跨族泄漏
  （`tests/test_companion_trajgen_pipeline.py`）。10^4 规模留给 LLM 模式批量跑
  （管线已就绪，规模由生成预算决定）。
- **M3 live-through 公版语料分支**：未启动（非 M1 范围）。M2（encoder 训练）未启动，
  G1-G4 gate 全部未过——不发布任何权重，对外口径不变。

## 实施记录（2026-07-18，M2 骨架落地）

训练/评测骨架落地为 `packages/companion-encoder/`（模块 `companion_encoder`，
Apache-2.0，核心依赖仅 `companion-standard`；torch 走 `[train]` extra，
transformers 走 `[hf]` extra）。代码开源，权重不随包发布（G1-G4 过审前无权重产物）。

- **数据接口只有轨迹 JSON**：`dataset.py` 读 companion-trajgen 输出布局
  （`<data_dir>/train|val/*.trajectory.json`），装载时重新过标准校验并
  强制整族 train/val 无交叠（交叠 fail-loud）。encoder 禁止 import
  `companion_bench` / `companion_trajgen` / 内核 wheel（AST 守门
  `tests/contracts/test_companion_encoder_boundaries.py`），也禁止引用
  `LabelSource`——它消费标签、绝不制造或洗白 provenance。
- **确定性序列化**（`serialization.py`）：anchor 前缀渲染，session 边界与
  gap 天数是一级文本（`[session k | gap Nd]`），因为 dormant/re_engaged
  等 phase 定义在多 session 结构上。
- **双头模型**（`model.py`）：backbone → phase 8 分类 + trust/continuity/repair
  回归（sigmoid）+ L2 归一 embedding 头。backbone 两档：`tiny`（自建
  byte-level transformer，CPU/MPS dry-run 用，明确非发布候选）与
  `hf:<model_id>`（Open Decision #1 的 fine-tune 路径，接缝已就位）。
  超长输入左截断保最近上下文。
- **G2 harness**（`evaluate.py` + `baselines.py`）：同一套纯 Python 指标函数
  给 encoder 与所有 baseline 打分——phase accuracy/macro-F1、逐项 MAE、
  ECE 校准、轨迹级 leave-one-out 检索（family top-1，对照标准 stub
  embedding 地板）。baseline 两列：majority/global-mean + LLM zero-shot
  （OpenAI-compatible 端点，prompt 集中在 `prompts/zero_shot_labeler.md`，
  不可解析输出记 `valid=False` 按中性值计分，不缩分母）。
- **接缝兑现**：`backend.py` 的 `EncoderEmbeddingBackend` 把训练 checkpoint
  包成标准 `SemanticEmbeddingBackend`（维度截断/补零 + 重归一，空文本
  委托 stub），`isinstance` 协议测试通过。
- **CLI**：`companion-encoder train / evaluate / baseline`；baseline 子命令
  torch-free（模块级禁 import torch，AST 守门）。
- **dry-run 验收**（90 条 FSM 轨迹，327 train / 183 val anchor 样本）：
  tiny backbone MPS 训练 3 epoch ≈ 3.7 分钟，全链路出 G2 报告。结果符合
  预期且诚实记录：phase 上 tiny 尚不敌 majority baseline（acc 0.45 vs
  0.75——FSM 模式表面文本是合成填充、90 条轨迹本就不含足够信号），
  检索已胜 stub 地板（0.67 vs 0.43）。G2 的真实较量在 M1 LLM 模式
  数据 + hf backbone 之后；止损逻辑不变。
- Apache header 守门（`test_apache_license_header_present.py`）已扩展覆盖
  本包。mirror 同步脚本**未**收编 encoder——公开发布是 M5/G4 之后的动作。

## 变更日志

- 2026-07-18 v0 草稿。与 Option A spec 同批产出；训练部分为规划性内容，
  代码落地时以代码为准并回改本 spec。
- 2026-07-18 M0 + M1-v0 落地（companion-trajgen），新增实施记录一节。
- 2026-07-18 M2 骨架落地（companion-encoder：数据装载 + 双头模型 + 训练循环 +
  G2 harness + baseline + backend 接缝 + CLI），dry-run 全链路验证，新增实施记录一节。
