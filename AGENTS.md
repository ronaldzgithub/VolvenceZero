# Volvence Repository Agent Instructions

本文件是本仓库所有编码 Agent 的仓库级约束，作用于仓库根目录及全部子目录。关键词“必须 / 禁止”均为硬约束。`.cursor/rules/*.mdc` 是这些约束的来源和补充说明；若本文件未覆盖某个任务特例，必须继续读取对应的 Cursor rule。

## 1. 开始任何编码任务前

1. 先执行 `git status --short`，识别并保留用户已有改动。不要覆盖、回退或顺手整理无关文件。
2. 按以下顺序由近及远获取上下文：
   - `docs/specs/00_INDEX.md`：定位能力域、目标 wheel 和 owner；并核对任务落在哪条能力轴（Appendable / Readable / Learnable / Steerable）。
   - 对应 `docs/specs/*.md`：确认职责边界、契约和不变量。
   - 仅当涉及跨域边界、接口或架构意图时，再读 `archetecture.md`、`docs/next_gen_emogpt.md` 与 `docs/appendable-readable-learnable-steerable.md`。
   - 最后读取具体实现和测试。
3. 文档与代码冲突时，以当前代码行为为准，并在同一改动中修正对应 spec。
4. 先定位根因所属能力域和唯一 owner。功能设计、行为逻辑、决策流程不得用表达层补丁掩盖上游问题。参数校验、日志格式、拼写等简单 bug 可直接局部修复。
5. 跨模块或跨库新增行为，必须先在 `docs/DATA_CONTRACT.md` 的 slot 注册表明确 `owner / value_type / dependencies / wiring_level`。
6. 改动前用四能力轴自检（见 §2.1）；任一条答不上来，不得用 prompt/关键词/evaluation 回灌冒充系统能力。

## 2. 系统设计的不可让步原则

Volvence 是融合 NL（Nested Learning）与 ETA（Emergent Temporal Abstractions）的有界、契约驱动、持续适应系统。核心产品是关系与主体性（EQ + 信任），不是单纯任务智力。

- 系统不是“静态模型 + prompt”，也不只在 token 层学习。
- 记忆、优化、控制是同一自适应系统的不同层，不得割裂。
- 明确区分 `online-fast / session-medium / background-slow / rare-heavy`；快速适应位于有界控制器，冻结基底只离线或极慢更新。
- 长期策略学习位于控制器代码 `z_t` 空间；禁止 token 空间 RL，禁止在线端到端更新整个基础模型。
- Prediction Error 是一级原始信号；evaluation、credit、needs、homeostasis 都只能是其下游 readout。evaluation 不得反向成为学习源。
- World / Self 双轨语义隔离；关系连续性不是完成任务的副作用。
- regime 是可记忆、可选择、可训练的运行时身份，不是 prompt 标签。
- 在线控制器更新与 rare-heavy artifact 更新分层；后者必须经过 `ModificationGate`，不得 bypass。
- 新自适应层必须有唯一 owner、可检查公共交换、明确退出条件、评估证据和可回滚部署。

### 2.1 四能力轴主张（Appendable · Readable · Learnable · Steerable）

系统主张必须同时满足四条能力轴；缺任一轴只能声称机制局部成立，不得对外说成「在线持续主动学习系统已成立」。完整说明见 `docs/appendable-readable-learnable-steerable.md`；索引入口见 `docs/specs/00_INDEX.md` 顶部「系统主张」节。

| 能力轴 | 必须成立 | 禁止塌缩为 | 主写者 |
|---|---|---|---|
| **Appendable** | 经历按时间尺度写入 CMS / State-KV / semantic 快照，可跨 session 恢复，不重写基底 | prompt 堆历史；单一 KV 冒充记忆 | `vz-memory`、substrate carriers、semantic owners |
| **Readable** | 内部状态从残差与不可变快照**命名读出**并发布 | 关键词/正则文本路由；consumer 重建 producer 隐状态 | `vz-substrate` residual / N+1 target、`prediction_error`、`steering_condition_belief` |
| **Learnable** | 只从 PE 及其下游 credit 学习；终局信用可走 PE→credit→gate | 用 evaluation / judge / 七日 continuity 分数当 reward | `vz-cognition` PE/credit/ModificationGate；`vz-temporal` gate |
| **Steerable** | 冻结基底上有界、条件化、可择时残差干预（norm cap、无 free bias、strict noop） | 端到端微调；token 空间 RL；ACTIVE 读 shadow fallback | sensor→executor→gate（`steering_*`，默认 SHADOW） |

闭环：Appendable → Readable → Learnable → Steerable → 下一拍可追加状态。
steering 三件套契约：`docs/specs/steering-runtime.md`。人类标注只作验证锚，不作学习源：`docs/specs/steering-human-anchor.md`。

改代码前必须能回答：

1. 写入落在哪个时间尺度、哪个唯一 owner？如何恢复？
2. 新状态是否以 frozen snapshot 发布？消费者是否只读快照？
3. 学习信号是否来自 PE/credit？有无 evaluation/judge 泄漏？
4. 干预是否有界且 lineage / WiringLevel 可单字段回滚？
5. 干预是否改变下一拍可追加状态，从而让 PE 可结算？

诚实边界：代理上「读得到 + 扳得动 + 学会何时扳」与 SHADOW runtime 接线不等于 production ACTIVE 或 thesis 通过。晋升路线见 `docs/moving forward/主线提升方案_2026-08.md`。

## 3. Wheel 边界与唯一职责

| Wheel | 唯一职责 |
|---|---|
| `vz-contracts` | Snapshot、RuntimeModule、Guards、propagate；所有 wheel 可依赖 |
| `vz-substrate` | 冻结 LLM、残差捕获、有界 adapter-delta 入口；不做策略、不做 prompt |
| `vz-temporal` | metacontroller（encoder、`beta_t`、decoder）和 `z_t` 上的 Internal RL |
| `vz-memory` | CMS 四层与 background-slow ReflectionEngine |
| `vz-cognition` | PredictionError、credit、ModificationGate、dual-track、regime、9 类语义 owner、evaluation |
| `vz-application` | domain knowledge、case memory、playbook、boundary policy；垂直经验编译进既有 owner |
| `vz-runtime` | 薄编排；唯一允许 import 其他业务 wheel 的业务层 |
| `lifeform-*` | 产品 / 生命体适配；只能经 Brain facade、contracts、`ModificationGate` 进入脑核 |

- 禁止 `vz-*` 反向 import `lifeform-*`。
- 仓库根目录的 `pyproject.toml` 只是 workspace meta-package；运行时代码必须进入 `packages/` 下的合适 wheel。
- 历史能力名 `vz-pe-credit / vz-self-model / vz-evaluation` 不是当前 wheel；对应实现位于 `vz-cognition` 子包。未来物理拆 wheel 时，必须同步修改 `docs/DATA_CONTRACT.md`、`archetecture.md`、`tests/contracts/test_import_boundaries.py` 和相关 `pyproject.toml`。

## 4. 快照、SSOT 与模块隔离

1. 快照是独立模块间唯一数据通道。模块 A 需要模块 B 的数据时，只能消费 B 发布的不可变快照，禁止持有、import 或直接调用 B。
2. 谁拥有数据，谁负责解释数据。描述、摘要和派生语义必须由 owner 生成并随快照发布；consumer 不得遍历内部结构、硬编码内部字段或重建 producer 状态。
3. 快照和 value 必须是不可变对象（优先 frozen dataclass）。禁止泄露内部可变引用、原地修改或用 `copy.deepcopy()` 模拟隔离；使用 `dataclasses.replace()` 和结构共享。
4. 快照缺字段时，去 publisher 丰富正式契约，不在 consumer 建立第二 owner。
5. 编排器只能传播和读取快照，不直接调用模块处理方法；系统初始化器可在启动时构造模块并注入底层组件。
6. 迁移使用 `WiringLevel.ACTIVE / SHADOW / DISABLED`：新旧并跑、比较快照、再切换，且必须可回滚。
7. 职责边界、输入输出、快照 shape、共享 schema、owner 或关键不变量变化时，必须同步更新 `docs/specs/*.md`；没有 spec 就在本次改动中补充。

九类语义状态 `plan_intent / commitment / open_loop / user_model / execution_result / belief_assumption / relationship_state / goal_value / boundary_consent` 各自保持唯一 owner，并通过快照发布。

## 5. 禁止补丁式决策逻辑

- 禁止用字符串包含、正则文本匹配、关键词字典等方式把自然语言映射为 emotion、scene、mode、route 或 action。
- 语义决策应使用嵌入相似度、LLM 结构化输出、分类模型或学习到的 `beta_t / z_t` 内部表示。
- 精确匹配协议枚举或 API 参数（如 `action == "stop"`），以及日志过滤、文本搜索等工具用途不受此限。
- 禁止用硬编码行为规则替代系统本应学习的能力。
- 禁止在表达层通过 if/else 或 prompt 技巧替代上游策略、信用分配和长期学习。

## 6. 错误处理与兼容性

- 契约违反必须 fail loudly。
- 禁止 bare `except:`。
- 禁止 `try/except` 后 `pass`、`continue`、空处理或无记录的静默回退。
- 禁止用宽泛 `hasattr()` 或 `getattr(obj, name, default)` 隐藏 schema 不匹配、拼写错误或版本漂移。
- 只捕获可预期的具体异常，并添加上下文后 re-raise，或执行已文档化且可观察的显式回退。
- `except Exception` 仅允许添加上下文后立即 re-raise，或位于明确的进程 / 请求故障边界且记录完整错误。
- 可选字段或能力必须由类型和契约明确表达；检查后分支必须清晰可测。
- Ruff 对 `S110 / S112 / E722` 零容忍。

## 7. LLM prompt 与 schema

- 大段 prompt 必须存放于专门的 `prompts/` 目录，并通过集中模板管理器加载和渲染。
- JSON Schema 必须存放于专门的 `schemas/` 目录，禁止在运行时代码内联大型 dict。
- LLM 是表达层和 background-slow 反思工具，不是 metacontroller、RL 信号、信用分配或长期策略 owner。
- 例外：不超过 5 行的 anti-loop guard / 单句 hint，以及测试中的 mock prompt 可以内联。

## 8. 大型架构改动：收敛包

多文件架构改动或模块边界重构必须拆成可解释、可回滚的收敛包：

- 一个包只解决一个 owner、冻结一个正式快照契约、切换一个主要 consumer，优先控制在 3–8 个关键文件。
- 基底层改动与控制器层改动不得放在同一包；CMS 不同频率层也应独立于基底改动。
- 高 ripple 的共享契约单独成包并后置。
- owner / publisher 与 consumer / cleanup 分开推进。
- 禁止一次替换整条主链、同时发明新 shape + 接完所有 consumer + 删除全部 fallback，或一次新增多个 owner。
- 每个包完成时检查：owner 唯一、正式交换只走契约、consumer 快照优先、legacy 路径已隔离/标记/删除、spec 已同步、退出与回滚条件明确、评估证据已产出。

若当前运行环境支持计划或并行 agent，先把任务收敛为一个包；只在用户或当前执行环境明确允许委派时使用子 agent。任何时候同一快照链只能有一个写入者。

## 9. 预训练任务的附加约束

预训练是 NL 最低频率层的 in-context learning：

- 预训练建立冻结或极慢更新的稳定基底；不得联合训练基础模型和 metacontroller。
- 复用线上正式模块的独立调用模式，仍遵守快照契约；禁止自建简化 reward、状态编码或路径选择。
- 奖励从真实经历经信用分配产生，回写需携带语义上下文和结果。
- 记忆写入必须走 memory owner 的正式 API，禁止直接构造卡片或访问模块内部字段。
- T=0 只提供根需求，认知结构随经历生长，不预设完整拓扑。
- 每个 chapter / episode 后的慢反思必须同时产出记忆整合与策略整合，且不得阻塞实时 turn。
- 遵循 SSL 压缩表示、再在 `z_t` 空间 RL 强化的交替关系。

## 10. Scenario Package 任务的附加约束

当任务要求生成 scenario package 时，必须遵循 `.cursor/rules/scenario-package-generation.mdc` 的完整格式，并至少满足：

- 必须生成 `manifest.yaml`、`ssot_fragment.json`、`test_suite.yaml`；推荐生成 `scenes.yaml`。
- `manifest.name` 匹配 `^[a-z][a-z0-9_]*$`，解释不少于 200 个中文字符，并覆盖路径、弧线、语义检测、集成点和 R14 体制身份。
- 每个 path 至少被一个 arc 引用；phase order 从 0 连续递增。
- 至少 6 个 routing tests（含 negative case）和 3 个 semantic coherence cases。
- 场景检测必须是语义级方法，禁止关键词路由。

## 11. UploadLive i18n

修改 `UploadLive/**/*.{tsx,jsx,ts,js}` 时：

- 所有用户可见文本必须使用现有 i18n `t(...)`，禁止中英文硬编码或按 `language` 手写条件文本。
- 新 key 使用 camelCase、按功能域分组、使用描述性名称。
- 在 `UploadLive/src/i18n/translations.js` 同时补全 `zh / en / ja / es / ar / de / fr / ko / id / th` 十种语言。
- 参数化文案使用现有 `{placeholder}` 约定。

## 12. 实施与验证

- 优先修改既有 owner 和契约，不复制平行实现。
- 测试应覆盖根因、契约边界、失败路径和回滚 / wiring 行为；不要只断言表面输出。
- 每个收敛包默认只运行与其根因和影响面直接相关的验证，不得仅因“属于收敛包”、改动文件较多或跨多个目录就默认运行全仓测试。常用命令：
  - `ruff check <changed paths>`
  - `pytest <relevant test files>`
  - 一般跨 wheel 变更先运行直接相关的 contract tests
  - 修改共享快照 / schema、wheel 依赖或 import boundary、全局 wiring / 初始化路径时，追加 `pytest tests/contracts`
- 全仓 `pytest`（默认跳过 `live_network`）不是单个收敛包的默认完成条件，仅在以下情况运行：
  - 用户明确要求；
  - 发布、主分支合并或里程碑验收，且 CI 不会提供等价的全量回归；
  - 修改公共基础设施、全局初始化或其他无法可靠圈定影响范围的高 ripple 路径；
  - 相关测试暴露出疑似跨域回归，需要扩大范围定位。
- 模型、GPU、外部 API、长轨迹、多 seed 或 repeated-run evidence 不属于普通全量回归。仅当对应机制、算法变量、证据契约或 promotion gate 改变时运行；普通重构、文档或非算法字段修改不得触发重复的昂贵 evidence run。
- 不得为让测试通过而降低断言、吞异常、添加关键词 hack 或恢复非正式 fallback。
- 交付时说明：修改的 owner / 契约、运行过的验证、未运行项及原因（包括未运行全量测试的原因）、迁移退出和回滚方式（如适用）。

## 13. Git 提交说明

只有用户明确要求提交时才创建 commit。提交前必须重新检查暂存区，确保不混入用户改动或与本任务无关的文件。

- commit message 的标题和正文必须使用中文；代码标识符、文件名、协议名等无法合理翻译的专有名词可以保留英文。
- 禁止只写 `fix`、`update`、“修复问题”、“更新代码”等无法独立说明改动的空泛信息。
- 标题应准确概括“改了什么以及目的是什么”，使用明确的动宾结构，避免罗列文件名。
- 除纯拼写修正等极小变更外，必须填写详细正文。正文应使未参与开发的人仅阅读提交记录，就能理解变更动机、实现边界和验证结论。
- 正文根据实际改动完整说明：
  1. **背景与问题**：原有行为、触发条件、用户影响或架构缺口。
  2. **根因分析**：问题所属能力域、唯一 owner，以及为什么原实现不满足契约或设计不变量。
  3. **修改内容**：关键实现、数据流、契约 / 快照、consumer 和文档如何变化。
  4. **兼容与迁移**：兼容性影响、`WiringLevel`、legacy 路径、退出条件和回滚方式；不适用时明确说明。
  5. **验证结果**：实际执行的测试、lint、关键场景和结果；未执行的检查及原因必须如实记录。
  6. **风险与后续**：已知限制、残余风险或需要后续处理的工作；没有时明确写“无已知遗留风险”。
- 提交说明只能描述本 commit 实际包含的内容，禁止声称未执行的测试、未完成的迁移或未落地的行为。
- 一个 commit 只承载一个清晰意图。若改动包含互不相关的目标，应拆分提交，并为每个提交分别提供完整中文说明。

推荐格式：

```text
<类型>：<用中文准确概括改动及目的>

背景与问题：
- ...

根因分析：
- ...

修改内容：
- ...

兼容与迁移：
- ...

验证结果：
- ...

风险与后续：
- ...
```

## 14. 权威参考

- `docs/specs/00_INDEX.md`：默认知识入口（含四能力轴主张与能力域映射）
- `docs/appendable-readable-learnable-steerable.md`：Appendable / Readable / Learnable / Steerable 完整架构说明
- `docs/DATA_CONTRACT.md`：快照 schema、slot 注册表、依赖图、变更协议
- `docs/next_gen_emogpt.md`：R1–R15、R-PE 与 NL / ETA 算法依据
- `archetecture.md`：wheel 切分、边界和迁移路线
- `docs/prd.md`：愿景、工程拆解和 M0–M6
- `SPLIT.md`：仓库边界 charter
- `docs/moving forward/主线提升方案_2026-08.md`：机制证据 → 系统主张的晋升路线
- `.cursor/rules/*.mdc`：本文件的细化规则和任务模板
