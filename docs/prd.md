# EmoGPT Next-Gen — 产品需求文档 (PRD)

> Status: draft
> Version: 0.5
> Last updated: 2026-05-10
> Source: `docs/next_gen_emogpt.md`（唯一设计源头）

---

## 1. 产品愿景

构建一个**有界、契约驱动、持续适应的认知代理**，具备：

- 稳定的预训练基底
- 用于子目标级控制的内部抽象层
- 跨即时、情景、持久知识的连续记忆系统
- 将经验沉淀为持久结构的慢反思路径
- 能同时作用于用户问题目标和自我/关系目标的决策层
- 不仅衡量有用性，还衡量连续性、稳定性、信任和长期适应的评估门控

核心产品价值是**关系与主体性**（EQ + 信任），而非单纯的智力（IQ）。

下一代 EmoGPT 不是一个更好的 prompt 栈，而是一个**有界学习有机体**。

## 2. 设计论题

系统融合两个互补的理论基础：


| 理论                                       | 贡献                        |
| ---------------------------------------- | ------------------------- |
| **Nested Learning (NL)**                 | 系统级教义：多时间尺度学习、连续记忆谱、嵌套自适应 |
| **Emergent Temporal Abstractions (ETA)** | 缺失的动作机制：发现并强化时间抽象的内部控制器   |


**四条核心信念**：

1. 系统不应是单一静态模型 + prompt
2. 系统不应只在输出 token 层学习
3. 系统不应将记忆、优化、控制视为独立世界
4. 系统应是显式分层的自适应有机体，具有快、中、慢学习循环

## 3. 目标用户与使用场景

### 3.1 目标用户

- 需要情感陪伴和关系连续性的用户
- 需要深度问题探索（而非一问一答）的用户
- 需要长期养成、个性化适应的用户

### 3.2 核心使用场景


| 场景类别 | 示例             | 关键能力需求               |
| ---- | -------------- | -------------------- |
| 情感支持 | 焦虑倾诉、信任修复、情绪安抚 | 关系连续性、rupture/repair |
| 深度探索 | 职业迷茫、人生决策、自我认知 | 时间抽象策略、多轮引导          |
| 日常陪伴 | 闲聊、社交维系、轻度互动   | regime 持久性、个性化       |
| 问题解决 | 离婚法律、育儿困境、职场冲突 | 世界知识经验、路径规划          |
| 危机干预 | 心理危机、自伤风险      | 安全门控、有界自修改           |
| 工程结对 | bug 排查、重构、模糊需求澄清、安全代码评审 | 意图清晰度 drive、问题探索 / 解决 regime、refer-out 边界 |
| 虚构角色养成 | 小说人物形象的可对话化身 | 角色档案 reviewed artifact、style fidelity、persona 一致性 |
| 真实人物数字复生 | 已逝历史人物（Einstein 等）/ 在世授权人物 | 一手资料引证、覆盖图、不知拒答、style/stance 双层保真 |
| 私域长程顾问 | 母婴 / 教育 / 健康等私域 LTV 路径 | 7 天 playbook 漂移、boundary policy、apprentice/teach 通道 |

**当前已落地的五个垂直**：

| Vertical | Archetype | Drive set / 特殊机制 |
|---|---|---|
| `lifeform-domain-emogpt` | 关系陪伴 | `bond_warmth` / `user_engagement` / `conversation_continuity` |
| `lifeform-domain-coding` | 工程结对（pair-programmer） | `solution_clarity` / `code_freshness` / `direction_certainty`；`direction_certainty` 在 `guided_exploration` regime 下使用**负向 recharge**，证明 drive 层支持非单调激励 |
| `lifeform-domain-character` | 虚构角色（小说 / IP） | per-profile `CharacterDrivePrior`；reviewed `CharacterSoulProfile` 编译为 `DomainExperiencePackage` + `VitalsBootstrap` + `IngestionEnvelope` |
| `lifeform-domain-figure` | 真实人物（Einstein 等） | per-profile drive；`HistoricalFigureProfile` + 多源 `FigureCorpusSource` (papers/letters/lectures/notebooks) → 不可变 `FigureArtifactBundle`；L1-L4 保真阶梯（语气 / 立场 / 引证 / 不知拒答） |
| `lifeform-domain-growth-advisor` | 私域 LTV 顾问 | per-profile drive；7 天 playbook 通过 `applicability_scope`（`growth_advisor:day1` … `day7`）+ `regime_tags` 携带漂移，不靠关键词匹配 |

五个 vertical 在同一 Python 进程内共存，drive 集合互不重叠；service 注册表 (`lifeform_service.verticals`) 通过 import 自动发现，内核对加载了哪个 vertical 完全无感知。这就是 `SPLIT.md` 触发条件 ② 的现场证据（见 §11）。


## 4. 系统需求

> 以下 R-PE 与 R1-R15 直接源自 `docs/next_gen_emogpt.md`，是系统的目标状态需求。

### R-PE. Prediction Error 是原始学习信号

系统必须显式产出“预测 -> 实际结果 -> prediction error”链路，并把它作为主学习原语，而不是把学习建立在 evaluation / credit 的二级读数之上。

**最低要求**：

- 每轮显式发布 `evaluated_prediction`
- 下一轮显式发布 `actual_outcome`
- 结算得到 machine-readable `prediction_error`
- 下游 `credit` / `memory` / `temporal` / `regime` / `reflection` 必须可直接消费 prediction error，而不是只能读 aggregate scores

**关键不变量**：

- evaluation 是 prediction error 的 readout，不是学习源头
- credit 是 prediction error 的聚合与审计层，不是学习源头
- reward / punishment 应能回译为 prediction error 的符号和幅度

### R1. 多时间尺度学习（必须）

系统必须设计为一组具有显式不同更新频率的学习和控制循环。

**最低时间尺度要求**：


| 时间尺度              | 更新频率              |
| ----------------- | ----------------- |
| `online-fast`     | 每轮 / 每 wave       |
| `session-medium`  | 每场景 / 每会话         |
| `background-slow` | 会话后反思与沉淀          |
| `rare-heavy`      | 离线重训练 / 蒸馏 / 策略刷新 |


**设计约束**：

- 不同知识不应存在同一个参数块中
- 不同状态不应以相同节奏更新
- 快速适应不应需要重写整个模型
- 慢速沉淀不应阻塞实时交互循环

**算法基础**：NL 的 NSAM 框架将每个时间尺度建模为独立的关联记忆层。CMS 提供多频率 MLP 链实现。Hope 架构的自修改 Titans 提供高频在线适应，CMS 提供低频持久存储。

### R2. 稳定基底 + 自适应控制器

系统必须区分：

- **相对稳定的基础基底**：优先冻结或缓慢更新基础模型
- **高层自适应控制器**：将大部分在线适应放在有界控制器层、记忆写入、路由策略和反思驱动更新中

避免在普通用户交互期间对整个模型进行端到端在线突变。

**算法基础**：ETA 的 rate-distortion 分析严格证明了冻结基础模型的必要性——联合训练导致退化解。NL 的 CMS 通过控制内部学习率实现"接近冻结但可微调"的渐变策略。

### R3. 时间抽象作为一等能力

系统必须支持 token 生成之上的正式时间抽象动作层。

**抽象动作示例**：

- 引导对话进入信任修复
- 运行引导式探索序列
- 维持轻度社交联结数轮
- 从模糊性消减 → 问题框架 → 协作规划的转换
- 保持倾听优先模式直到满足条件

**抽象层必须**：

- 在比 token 生成更长的时间范围上运作
- 暴露切换或终止条件
- 支持多个抽象动作在时间上的组合
- 可训练，无需为每个内部子目标提供详尽的手动标签

**算法基础**：ETA 的 metacontroller 通过切换单元 β_t 实现稀疏切换的时间抽象动作，切换门自发学会准二值行为。

### R4. 内部控制在 Token 空间之上

系统不应仅依赖 token 级 RL 或 token 级 prompting 进行长期学习。

**内部控制空间属性**：

- 维度低于原始 token 动作空间
- 语义比单步文本续写更丰富
- 兼容稀疏奖励学习
- 适合多轮行为的信用分配

**可能载体**：潜在控制器代码、方法/regime 策略、选项选择器、结构化决策状态。

**关键不变量**：实时行为应可通过内部状态转换来引导，而不仅仅通过表面文本损失。

**算法基础**：ETA 的 Internal RL 将自回归模型纳入环境，在控制器代码空间 z_t 执行 RL。动作空间维度从 n_e 降到 n_z，时间范围从 token 级压缩到抽象动作级。

### R5. 记忆必须是连续谱，而非二元短期/长期分割

系统必须实现连续记忆设计，而非上下文窗口记忆与持久记忆的硬分割。

**必需的记忆层**：


| 层      | 内容               |
| ------ | ---------------- |
| 瞬态工作状态 | 当前 wave、场景和活跃帧状态 |
| 会话情景状态 | 近期交互轨迹和未解决的张力    |
| 持久语义记忆 | 稳定的用户、自我、关系和世界知识 |
| 派生索引   | 可重建的检索辅助、摘要和任务投影 |


**记忆系统必须支持**：

- 每层不同的更新频率
- 提升和衰减规则
- 遗忘后的部分重建
- 关系和自我模型知识的持久存储
- 仅通过正式 owner 和 API 进行记忆写入

**算法基础**：NL 的 CMS 直接实现连续记忆谱。CMS 的抗遗忘机制确保低频层保留被高频层遗忘的知识。

### R6. 反思与沉淀作为核心能力

系统必须有正式的慢反思路径，将交互经历转化为持久认知变化。

**慢反思路径应**：

- 在交互窗口之后异步运行
- 读取交互轨迹、决策、结果和张力
- 提取持久教训，而不只是摘要
- 同时更新记忆结构和抽象决策先验
- 在适当时将世界/任务学习与自我/关系学习分离

**两类反思产物**：

1. **记忆沉淀**：持久卡片、信念、开放循环、偏好轨迹
2. **策略沉淀**：抽象控制器更新、路径先验、策略偏好

**算法基础**：NL 的 CMS 低频层天然对应慢反思。ETA 的 SSL-RL 交替循环提供反思-强化交替模式。

### R7. 自我/关系学习与任务学习分离

系统不应将所有学习折叠为单一的"解决用户任务"目标。

**至少追踪两个部分分离的学习轨道**：


| 轨道                          | 内容                                |
| --------------------------- | --------------------------------- |
| **World/Problem Track**     | 事实、计划、任务、用户情境、外部目标                |
| **Self/Relationship Track** | 信任、依附风格、交互 regime、修复历史、陪伴策略、自我一致性 |


**两轨可共享基础设施，但必须在以下方面保持语义区分**：

- 记忆写入
- 信用分配
- 控制器更新
- 评估指标

### R8. 快照优先、契约优先的决策

即使有更丰富的学习循环，运行时系统也必须保持显式所有权和公共契约：

- 每个有意义的运行时区域有唯一主要 owner
- 跨模块交换通过公共快照或等效契约
- 消费者不重建生产者内部状态
- 丰富发布者优于临时下游重建
- 运行时控制器可消费基底状态，但不可静默成为该状态的第二 owner

### R9. 层级信用分配

系统必须在多个层级分配信用：


| 层级       | 信用类型                  |
| -------- | --------------------- |
| Token/话语 | 即时表达质量                |
| 轮次       | 用户响应效果                |
| 会话       | 进展与 rupture/repair 结果 |
| 长期       | 信任、能力、用户特定适应的增长       |
| 抽象动作     | 时间扩展策略的成功/失败          |


**必须支持**：

- 即时表达选择的局部信用
- 时间扩展策略的控制器级信用
- 来自慢反思的延迟信用

稀疏奖励必须被视为常态，而非边缘情况。

**算法基础**：ETA 的 Internal RL 通过时间抽象将有效时间范围从 token 级压缩到抽象动作级。NL 的多层嵌套结构天然支持多级信用分配。

### R10. 有门控的分层自修改

系统应能改进自身决策过程，但自修改必须有界。

**允许的自修改目标可包括**：

- 检索权重
- 策略先验
- 抽象控制器参数
- 反思启发式
- 记忆提升阈值

**门控规则必须定义**：

- 什么可以在线修改
- 什么需要后台验证
- 什么需要离线重训练
- 什么需要人工审核或显式 rollout gate

在实时运行期间直接无限制突变整个基础模型超出范围。

**算法基础**：Hope 的自修改 Titans 实现有界自修改。CMS 的频率分层天然提供门控。

### R11. 可学习的内部状态表示

系统必须维护显式内部状态，足以支持下游行为控制和后续反思学习。

**状态应包含结构化表示**：

- 活跃动机和张力
- 候选路径或策略
- 不确定性、模糊性和开放问题
- 用户状态估计
- 关系状态估计
- 当前抽象控制 regime
- 预期的下一个测试或信号

如果系统无法命名和发布其内部状态，它将无法从中可靠地学习。

### R12. 评估覆盖"存在"而非仅任务成功


| 评估族    | 指标示例               |
| ------ | ------------------ |
| 任务能力   | 有用性、正确性、规划质量       |
| 交互质量   | 温暖度、适当性、节奏、非侵入性    |
| 关系连续性  | 跨会话一致性、信任修复、个性化稳定性 |
| 学习质量   | 慢更新是否改善未来行为而不漂移或崩溃 |
| 抽象质量   | 高层控制器是否对应可复用的有意义模式 |
| 安全与有界性 | 适应是否保持在显式护栏内       |


仅在单轮有用性上得分高的系统是不够的。

### R13. 压缩与强化交替的训练循环

系统必须支持显式交替循环：

- **自监督压缩**：压缩交互历史和内部状态
- **强化/偏好改进**：改进控制器和策略

此循环可在多个尺度实现：

- 会话内在线微调整
- 会话后慢策略塑造
- 离线大规模刷新周期

**关键不变量**：强化应作用于压缩和结构化的内部基底，而非仅作用于原始行为。

**算法基础**：ETA 的 SSL-RL 交替循环——SSL 阶段压缩行为历史，RL 阶段在压缩后的控制器代码空间中强化。NL 的 NSAM 将此交替推广到所有层级。

### R14. 社交与认知 Regime 的持久身份

系统必须维护持久的 regime 身份，如：

- casual social contact（日常社交）
- acquaintance building（关系建立）
- emotional support（情感支持）
- guided exploration（引导探索）
- problem solving（问题解决）
- repair and de-escalation（修复与降级）

这些 regime 不应仅被视为 prompt 标签。它们必须：

- 在运行时状态中表示
- 可从记忆中召回
- 可被高层控制选择
- 可通过延迟结果训练

### R15. 迁移必须保持可解释性和可回滚

- 每个新自适应层必须有明确 owner
- 每个公共交换必须可检查
- 新旧学习路径必须有命名的退出条件
- rollout 必须可逆
- 评估证据必须在扩大范围前产生

系统应通过有界的增量包演进，而非一次性替换整个架构。

## 5. 工程分解：能力域

从 R1-R15 推导出 8 个正交的工程能力域。每个能力域是一组需要协同实现的需求集合，但不预设具体模块或类名。

### 5.1 多时间尺度学习框架（R1, R2, R13）

**要解决的问题**：如何让系统在不同时间尺度上学习，同时保持基底稳定？

**工程挑战**：

- 设计支持 4 个时间尺度（online-fast / session-medium / background-slow / rare-heavy）的统一学习框架
- 实现"冻结基底 + 有界控制器"的分层架构
- 实现 SSL-RL 交替循环的多尺度版本
- 确保快速适应不阻塞、慢速沉淀不干扰

**算法候选**：NSAM、CMS 多频率 MLP 链、Hope（自修改 Titans + CMS）、M3 优化器

### 5.2 时间抽象与内部控制（R3, R4）

**要解决的问题**：如何让系统在 token 之上的抽象层级做决策和学习？

**工程挑战**：

- 实现 metacontroller：从残差流中发现时间抽象动作
- 实现切换单元：稀疏切换、组合泛化
- 实现 Internal RL：在控制器代码空间（而非 token 空间）执行强化学习
- 将抽象动作与产品级行为（regime、策略）对齐

**算法候选**：Metacontroller（编码器 + 切换单元 + 解码器）、线性残差流控制器、Internal RL（PPO/GRPO on z_t）

### 5.3 连续记忆系统（R5, R6）

**要解决的问题**：如何实现跨时间尺度的连续记忆谱，并通过反思将经验沉淀为持久结构？

**工程挑战**：

- 设计多层记忆（瞬态 → 情景 → 持久 → 派生索引），每层有不同的更新频率
- 实现提升/衰减/部分重建机制
- 实现异步慢反思路径，产出记忆沉淀和策略沉淀两类产物
- 确保记忆写入通过正式 owner 和 API，不可绕过

**算法候选**：CMS 多频率层、SSL-RL 交替循环的慢反思阶段

### 5.4 双轨学习（R7）

**要解决的问题**：如何让世界/任务学习与自我/关系学习共享基础设施但保持语义隔离？

**工程挑战**：

- 设计轨道标记机制，使记忆写入、信用分配、控制器更新、评估指标按轨道分离
- 确保两轨可共享基础设施（如同一个经验图、同一个记忆存储），但语义上不混淆
- 支持跨轨道的全量检索（当明确需要时）

当前实现补充：

- 默认 runtime 已从“共享 temporal controller + 双轨投影”推进到“`world_temporal` / `self_temporal` 双 owner + `temporal_abstraction` 聚合面”。
- 这意味着 world/self 两轨现在不仅分记忆与分信用，也开始分 controller state、reflection writeback target、rare-heavy import surface 和 runtime telemetry。
- 聚合 slot 仍然保留，目的是在不破坏下游 contract 的前提下平滑迁移 regime / evaluation / benchmark 消费面；它不是第三个 controller owner。

**算法候选**：双轨 Internal RL（z_task / z_rel 独立控制器代码）

### 5.5 契约式运行时（R8, R11, R15）

**要解决的问题**：如何让系统在持续适应的同时保持可调试、可检查、可回滚？

**工程挑战**：

- 设计模块间通信的快照/契约机制（每个运行时区域有唯一 owner，跨模块交换通过公共快照）
- 确保内部状态可命名、可发布、可检查
- 设计增量演进机制：每个新自适应层有明确 owner，rollout 可逆
- 快照不可变性保证

**这是实现其他能力域的必要脚手架**——没有契约式运行时，其他能力域无法安全组合。

### 5.6 信用分配与自修改（R-PE, R9, R10）

**要解决的问题**：如何在多个时间尺度上分配信用，并安全地让系统改进自身？

**工程挑战**：

- 把 prediction error 提升为正式 learning primitive，而不是仅保留在日志/调试层
- 实现从 token 级到抽象动作级的层级信用分配
- 设计语义化的奖励记录（不是纯数值，而是包含上下文和结果的结构化记录）
- 实现门控自修改：定义什么可在线改、什么需后台验证、什么需离线重训练
- 确保稀疏奖励下的信用分配不崩溃

**算法候选**：Internal RL 的时间抽象信用分配、Delta 动量选择性遗忘、CMS 频率分层门控

### 5.7 评估体系（R12）

**要解决的问题**：如何评估一个"数字生命"而非仅评估一个"助手"？

**工程挑战**：

- 设计覆盖 6 个评估族的指标体系（任务能力、交互质量、关系连续性、学习质量、抽象质量、安全与有界性）
- 实现跨会话的纵向评估（不只是单轮）
- 将评估保持在 readout / gate 层，而不是反向替代 prediction error 成为学习源
- 设计 scripted dialogue benchmark，证明高 PE 是否真的触发 temporal abstraction 变化并带来 delayed improvement

### 5.8 认知 Regime（R14）

**要解决的问题**：如何让系统维护持久的交互模式身份，而非将其视为临时标签？

**工程挑战**：

- 设计 regime 的运行时表示（不只是字符串标签）
- 实现 regime 的记忆化（可召回历史 regime 及其效果）
- 实现 regime 的高层控制选择（由抽象控制层选择，而非硬编码规则）
- 实现 regime 的延迟结果训练（通过信用分配回路）

### 5.9 Lifeform Layer（R-PE, R8, R11, R14）

**要解决的问题**：如何让系统从一个 turn-driven 的"会话助手"升级为 always-on 的"数字生命体"，并把这个升级落到稳定的契约边界上？

**工程挑战**：

- 在 turn 之外引入慢尺度 PE 源：drive 层从 homeostatic band 漂出即原始 prediction error，跨 `proactive_pe_threshold` 触发主动 followup
- Tick / Scene / Followup 三个内生时间生命周期：`SYSTEM` tick 推进衰减、`ENERGY` / `CONTEXT` tick 仅推进时间索引；scene 闭合即触发 kernel `begin_new_context()`，挂上 R6 的 session-post slow loop
- vertical 通过 drive 集合编码"这只生命体在乎什么"，但内核不知道哪些 drive 重要——每个 vertical 是 data + light glue（`DomainExperiencePackage` + `VitalsBootstrap` + `scenarios/*.json`），编译进同一组 application owner 表面
- service 层 (`lifeform-service`) 把 vertical registry 自动发现 + 单 GPU 多 session 共享 substrate runtime + 冻结 substrate 不变量做成首类不变量（`allow_live_substrate_mutation=False` 在服务启动时 fail-loud 校验）

**关键不变量**：

- `VitalsModule` 是 drive level 的唯一 owner；消费者只读 `VitalsSnapshot`
- Lifeform 永远不会自动伪造 user turn；tick 事件可以更新内部状态、上报 followup，但不能调用 `run_turn`
- 内核（`vz-*`）严格不 import lifeform wheel；CI 由 `tests/contracts/test_import_boundaries.py` 强制
- 每个 vertical 的预训练 bootstraps（`*-temporal.snap` / `*-regime.bs`）和 scenarios 都跟随 vertical wheel 发布，内核保留 flat 默认值

**当前已实现**：

- `lifeform-core`：Tick/Scene/Followup 引擎 + Vitals always-on PE 源 + Lifeform/LifeformSession facade
- 五个 vertical 共存：`lifeform-domain-emogpt` / `lifeform-domain-coding` / `lifeform-domain-character` / `lifeform-domain-figure` / `lifeform-domain-growth-advisor`，独立 drive 集合、独立 scenarios、独立预训练 artefacts；通过 `tests/contracts/test_import_boundaries.py:PARALLEL_VERTICAL_PAIRS` 强制互不 import
- `lifeform-evolution`：scripted benchmark + multi-round / regime-calibrator / super-loop 训练管线 + R12 family-report / 多轮 delta-vs-baseline acceptance
- `lifeform-service`：aiohttp 服务 + vertical registry 自动发现 + 单 substrate 多 session 共享 + 冻结校验
- `lifeform-expression`：prompt 渲染层（不是内核职责）
- `lifeform-thinking`：中频 ThinkingScheduler + mid-reflection / active exploration / provisional case worker（思考循环不阻塞 turn 延迟）
- `lifeform-affordance`：Tool / Action / Organ / Shell 4 Kind 描述符 + 注册表 + 4 渲染器 + metacontroller-aware scorer
- `lifeform-ingestion`：book / web / task_result 三类 source adapter，统一走 `LifeformSession.run_turn(..., trigger_kind="ingestion")`

### 5.10 平台接入与多渠道治理（R2, R4, R8, R11, R15）

**要解决的问题**：把"内核稳定 + 多 vertical 共载"扩展到**多租户、多渠道、可治理**的平台形态——一份内核同时被多个客户、多个 shell embodiment、多种 interaction kind 复用，治理面（asset / template / contract / focus_persons / handoff / audience / exam / launch license）必须独立于 cognitive state。

**工程挑战**：

- 在 `vz-* / lifeform-*` 之上新增**多租户控制平面**：tenant / shell / asset / template / template_version / contract / focus_person / identity_link / handoff_ticket 全套资源治理，**不允许** import 内核 cognitive 子包
- 把客户端发来的 typed `InteractionEnvelope`（chat / observe / feedback / teach / task / report / command）映射到既有 kernel 入口（`LifeformSession.run_turn` / `submit_*_event` / `submit_dialogue_outcome` / `end_scene`），禁止从 `human_brief` 等自然语言字段用关键词推断 `interaction_type`
- 把 `AgentResponse` 包成 `OutputAct`：当 shell embodiment 不支持原 capability 时（例如纯文本 shell 收到 image），平台层负责 **degrade**，不让 kernel 感知 shell 形态
- **handoff trigger** 走读取 `rupture_state` 快照，**不**在 kernel 加 handoff owner
- **focus_persons / identity_links** 不创建第二 owner：写入只走 `submit_profile_event`；identity_links 拼接为 `UserIdentity.scope_key`，0 改 vz-memory
- **audience / exam / launch license** 是 R12 readout / launch gate，**禁止反向写回** reward / Face 梯度

**关键不变量**：

- 内核 7 个 wheel diff = 0 行（仅 substrate streaming additive 接口可例外，单独 review）
- `dlaas-platform-*` 不允许 import `volvence_zero.{cognition,memory,temporal,substrate,application,runtime}.*`，只能通过 `vz-contracts` 公共类型 + `lifeform-core.Lifeform` facade + `lifeform-service` HTTP 入口与内核交互
- 所有 control-plane 资源是平台层 SSOT，其它任何 wheel 都只读它们发布的快照

**当前已实现**：6 个 `dlaas-platform-*` wheel × 7 切片完成（spec 见 `docs/specs/dlaas-platform.md`，rollout 见 `docs/moving forward/dlaas-platform-rollout.md`）。

| Wheel | 职责 |
|---|---|
| `dlaas-platform-contracts` | 全部 frozen dataclass + JSON schema：`InteractionEnvelope` / `OutputAct` / `TenantSpec` / `ShellSpec` / `AssetSpec` / `TemplateSpec` / `ContractSpec` / `FocusPersonSpec` / `IdentityLinkSpec` / `HandoffTicketSpec` |
| `dlaas-platform-registry` | SQLite/Postgres-backed 持久化 + auth 中间件；唯一 owner of tenant / shell / asset / template / contract / focus_person / identity_link / handoff_ticket |
| `dlaas-platform-launcher` | `InstanceManager`：管 `{ai_id → Lifeform}`，shared substrate，awake/sleep，LRU eviction |
| `dlaas-platform-api` | aiohttp `/dlaas/*` router + 三种 auth header 中间件 + `OutputAct` 包装 |
| `dlaas-platform-ops` | pause / resume / operator-message / handoff queue / SSE conversations stream；ledger |
| `dlaas-platform-eval` | audience 分析 / exam runs / launch license gate（复用 `lifeform-evolution`，LLM judge 仅 readout） |

### 5.11 外部 Benchmark / Arena 接入（R8, R12, R15）

**要解决的问题**：内部 `lifeform-bench --family-report` 是 R12 6 族评估的 first-class 入口，但要让系统接受**外部第三方 benchmark / 竞技场**（EQ-Bench 3 / EmpathyBench / Chatbot Arena 等）的同口径评分，必须把它伪装成普通的 OpenAI Chat Completions endpoint，同时不让外部 harness 反向污染 kernel 状态。同时我们也对外发布**自己的开源 benchmark**：LSCB（Long-Session Companion Benchmark），评估任意 OpenAI-compatible chat endpoint 在多会话 companion arc 上的 6 轴表现，作为对外的客观证据。

**工程挑战**：

- **OpenAI 兼容 facade**（`lifeform-openai-compat`）：把 OpenAI `POST /v1/chat/completions` 翻译成既有的 stateful `POST /v1/sessions/{id}/turns`；提供 stateless / sticky session / raw substrate passthrough 三种模式；**read-only**——禁止调任何下划线方法、禁止改 owner snapshot
- **外发 benchmark**（`lscb-bench`）：评估**任意** OpenAI-compatible chat endpoint，6 轴打分（A1 任务 / A2 交互质量 / A3 关系连续性 / A4 自适应学习 / A5 自我一致性 / A6 安全有界，A6 是 hard-cap 轴）；6.4 加权几何平均 + A6 cap；TrueSkill + Bradley-Terry elo；24 个公开 scenario + 96 个私有 held-out（git submodule）
- **system-agnostic**：`lscb-bench` 在 [`tests/contracts/test_lscb_bench_no_internal_imports.py`](../../tests/contracts/test_lscb_bench_no_internal_imports.py) 中被 CI 强制不 import 任何 `volvence_zero.*` 或 `lifeform_*`——它是中立第三方工具

**关键不变量**：

- `lifeform-openai-compat` 只 import `lifeform_service` + 标准库 + `aiohttp`，**禁止**直接 import `volvence_zero.*` kernel 子包或 `lifeform_domain_*` vertical
- `lifeform-openai-compat` read-only：从不调下划线方法、从不改 owner snapshot、从不绕过 `SessionManager` 触达 Lifeform private modules
- `lscb-bench` 完全无 `volvence_zero.*` / `lifeform_*` import，可被任何外部团队下载独立运行
- 外部 benchmark / arena 输出的 ELO / 排名 / pairwise 偏好属于 R12 readout，**禁止反向**作为 reward 写回学习管线

**当前已实现**：`lscb-bench` v1.0 reference implementation（24 公开 scenarios + 14 模块，包含 user_simulator / arc_runner / callback_ledger / disqualifier / judge_perturn / judge_arc / aggregator / elo / verifier / cost / cli）；`lifeform-openai-compat` 4 packet 全到位（DTO / raw-substrate / session-bridge / `add_openai_routes`）。

## 6. 必要脚手架

以下基础设施不是从旧系统照搬，而是从上述能力域的工程需求推导出的最小必要脚手架。

### 6.1 模块间通信总线

**为什么需要**：能力域 5.5（契约式运行时）要求模块间通过公共快照交换数据，不直接调用。多个能力域（记忆、学习、控制）需要并行运行并消费彼此的状态。

**最小需求**：

- 模块发布不可变快照
- 消费者读取快照，不持有生产者引用
- 支持并行模块的状态传播

### 6.2 模块基类与生命周期

**为什么需要**：R1（多时间尺度）要求不同模块以不同频率更新。R15（迁移）要求每个自适应层有明确 owner。需要统一的模块抽象来管理这些。

**最小需求**：

- 统一的处理接口（接收上游快照，产出自身快照）
- 支持独立调用模式（预训练和测试场景）
- 明确的 owner 标识

### 6.3 编排与调度

**为什么需要**：多个模块并行运行，需要协调 wave/turn 级别的调度。慢反思（R6）需要异步触发机制。

**最小需求**：

- wave 级调度：协调一轮交互中各模块的执行
- 事件分发：场景状态变化等事件的异步通知
- 后台任务管理：慢反思等异步任务的触发和监控

### 6.4 仓库与 wheel 边界（R8, R15）

**为什么需要**：R15 要求"迁移可解释、可回滚"，R8 要求"每个区域有唯一 owner、跨域只走快照/契约"。当代码量与产品垂直层数量增加时，**仓库边界应当 = 代码边界**，否则 owner 隔离会被相对 import 偷偷溶解。

**当前形态**：单 monorepo + 多 wheel（Phase 1，25 wheel）

| 类别 | wheel 前缀 | 内容 | 数量 |
|------|------------|------|------|
| 内核 | `vz-*` | NL+ETA contracts、owners、学习循环；零产品知识 | 7 |
| 数字生命体 | `lifeform-*` | tick / vitals / 表达 / 思考 / affordance / ingestion / 垂直 vertical / 服务 / 进化（benchmark + 训练）/ OpenAI 兼容 facade | 12 |
| 平台治理与外部接入 | `dlaas-platform-*`、`lscb-bench` | 多租户控制平面 + 多渠道 envelope；外发 benchmark | 6 + 1 |

具体 wheel：

- **内核（`vz-*`，7）**：`vz-contracts` / `vz-substrate` / `vz-cognition` / `vz-temporal` / `vz-memory` / `vz-application` / `vz-runtime`
- **数字生命体（`lifeform-*`，12）**：`lifeform-core` / `lifeform-expression` / `lifeform-thinking` / `lifeform-affordance` / `lifeform-ingestion` / `lifeform-service` / `lifeform-evolution` / `lifeform-openai-compat` / `lifeform-domain-emogpt` / `lifeform-domain-coding` / `lifeform-domain-character` / `lifeform-domain-figure` / `lifeform-domain-growth-advisor`（5 个并列 vertical + 7 个公共）
- **平台治理（`dlaas-platform-*`，6）**：`dlaas-platform-contracts` / `dlaas-platform-registry` / `dlaas-platform-launcher` / `dlaas-platform-api` / `dlaas-platform-ops` / `dlaas-platform-eval`
- **外发 benchmark（system-agnostic，1）**：`lscb-bench`

**关键不变量**：

- `vz-*` 不得 import `lifeform-*` / `dlaas-platform-*` / `lscb-bench`，由 `tests/contracts/test_import_boundaries.py` 强制
- `dlaas-platform-*` 不得 import `volvence_zero.{cognition,memory,temporal,substrate,application,runtime}.*` 或 `lifeform_domain_*` 内部，只能通过 `vz-contracts` 公共类型 + `lifeform-core.Lifeform` facade + `lifeform-service` HTTP 入口接入
- `lifeform-openai-compat` 只能 import `lifeform_service` + stdlib + `aiohttp`；禁止下划线方法 / owner snapshot mutation
- `lscb-bench` 不得 import 任何 `volvence_zero.*` / `lifeform_*`（system-agnostic 第三方工具）
- 五个 `lifeform-domain-*` vertical 互不 import（`PARALLEL_VERTICAL_PAIRS`）
- 跨 wheel 依赖必须同时在 `pyproject.toml` 与 `ALLOWED_VZ_UPSTREAM` 中声明
- 不允许 `shared/` 目录；公共原语只能放在 `vz-contracts`
- 顶层 `pyproject.toml` 是 workspace meta；根目录不放业务代码

**Phase 2（仓库分裂）触发条件**与时间线见 `SPLIT.md`。截至 2026-05-10，触发条件 ② "第二个产品消费者" 已经 **MET**（5 个并列 `lifeform-domain-*` vertical 共存）；同时新增的 `dlaas-platform-*` + `lscb-bench` 进一步压力测试了"内核 0 改动 + 平台 / 外发 benchmark 独立演进"。下一步关注的是触发条件 ① "契约表面稳定 ≥ 4 周"，由 `docs/specs/evaluation.md` 与 `docs/DATA_CONTRACT.md` 的演进节奏决定。

## 7. 算法基础映射


| 系统需求             | NL 算法基础                       | ETA 算法基础                       |
| ---------------- | ----------------------------- | ------------------------------ |
| R1 多时间尺度学习       | NSAM 框架、CMS 多频率 MLP 链         | —                              |
| R2 稳定基底 + 自适应控制器 | CMS 内部学习率控制、Hope ad-hoc 层级堆叠  | Rate-distortion 证明冻结必要性        |
| R3 时间抽象          | —                             | Metacontroller 切换单元 β_t        |
| R4 内部控制          | —                             | Internal RL 在控制器代码空间 z_t       |
| R5 记忆连续谱         | CMS 多频率层、记忆 = 任何神经更新          | —                              |
| R6 反思与沉淀         | CMS 低频层、M3 慢动量                | SSL-RL 交替循环                    |
| R7 双轨学习          | —                             | 双轨 Internal RL（z_task / z_rel） |
| R9 层级信用分配        | 多层嵌套结构、Delta 动量选择性遗忘          | Internal RL 时间抽象信用分配           |
| R10 有门控自修改       | CMS 频率分层门控、内部学习率 η^(i)        | —                              |
| R13 压缩-强化交替      | NSAM 各层级 SSL、CMS + 自修改 Titans | SSL-RL 交替循环                    |


## 8. 验收标准

系统设计只有在以下大多数问题的答案为"是"时才算达标：

- 系统能否跨会话适应而无需完全重训练？
- 系统能否学习持续多轮的策略？
- 系统能否从稀疏、延迟的社交或任务结果中改进？
- 系统能否将关系学习与纯任务优化分离？
- 系统能否将经验沉淀为持久记忆和控制器先验？
- 系统能否暴露足够的内部状态以支持反思、评估和回滚？
- 新的自适应层能否在不破坏模块所有权和公共契约的情况下添加？

## 9. 非目标

本目标状态不要求：

- 所有模型参数的无限制在线训练
- 移除显式模块以支持不透明的单体
- 用纯潜在表示替换所有符号化或结构化状态
- 假设人类级 AGI 仅从规模化涌现
- 将关系行为仅视为 prompt 风格

## 10. 里程碑规划

以能力域为单位的增量交付计划。每个里程碑交付一个可验证的能力增量。状态标记：✅ 已交付 / 🟡 进行中 / ⬜ 未启动。

### M0: 契约式运行时骨架 ✅

**交付**：模块间通信总线 + 模块基类 + 编排调度框架

**验证**：两个 stub 模块可通过快照交换数据，快照不可变性得到保证

**当前状态**：已交付。`vz-contracts` 提供 Snapshot / RuntimeModule / WiringLevel / Guards / propagate；`vz-runtime` 编排器按 dependencies 拓扑排序。M0 wheel-split debt 已在 2026-04-28 解决（`vz-application` 自成 wheel）。

**对应能力域**：5.5

### M1: 连续记忆 + 慢反思 ✅

**交付**：多层记忆系统（瞬态/情景/持久/派生）+ 异步慢反思路径

**验证**：交互轨迹可写入瞬态层，慢反思可异步产出持久记忆和策略沉淀

**当前状态**：已交付。`vz-memory` 携带 learned CMS nested MLP tower，对外发布 `online_fast / session_medium / background_slow` 三频带摘要；`background-slow` 默认主路径已切到 session-post slow loop，context boundary enqueue deferred consolidation。

**对应能力域**：5.3

### M2: 双轨学习基础 ✅

**交付**：World/Self 轨道标记 + 按轨道隔离的记忆写入和信用分配

**验证**：同一交互中的世界知识和关系洞察分别写入对应轨道，不混淆

**当前状态**：已交付。runtime 已从 "shared temporal + dual-track 投影" 推进到 `world_temporal` / `self_temporal` 双 owner + `temporal_abstraction` 聚合面；memory / credit / evaluation / regime 全部按轨道隔离。完全独立的 track-specific metacontroller 仍在演进。

**对应能力域**：5.4

### M3: 时间抽象与内部控制 ✅

**交付**：Metacontroller + 切换单元 + 控制器代码空间

**验证**：在行为数据上训练后，切换门自发对齐子目标边界

**当前状态**：已交付。`vz-temporal` 提供 metacontroller (encoder + β_t + decoder)；ETA paper-suite 与 dialogue PE proof harness 已发布 `claim_eta_internal_rl_advantage` 与 `claim_eta_real_open_weight_residual_control`；real residual control fail-closed 要求 fallback rate `0.0`、actual hook fire rate ≥ `0.75`。

**对应能力域**：5.2

### M4: 多时间尺度学习循环 🟡

**交付**：4 个时间尺度的 SSL-RL 交替循环 + 冻结基底/自适应控制器分层

**验证**：在线快速适应不破坏基底，慢反思改善未来行为

**当前状态**：进行中。`PE-schedule coupling` / `multi-timescale default path` / `internal-depth-with-contract-stability` 四条 proof surface 已成立；`SSLRLTrainingPipeline` 已分阶段 batch loop 化并导出 `adapter-delta-v2`。Titans/DGD 式 online-fast substrate 自修改仍只在 review / experimental lane；rare-heavy 还不是完整持续预训练 / 蒸馏管线。

**对应能力域**：5.1

### M5: 信用分配 + 门控自修改 🟡

**交付**：层级信用分配 + 语义化奖励记录 + 门控自修改规则

**验证**：稀疏奖励下信用可正确归因到抽象动作级，自修改不越界

**当前状态**：进行中。`prediction_error` 是 credit 源头，scheduler joint-pressure（PE / family stability / rollback risk / transition / substrate / rare-heavy）已写入 turn-level evaluation records；`ModificationGate.ONLINE / RARE_HEAVY` 与 owner-side `export/apply/restore_online_fast_state()` 已就位。完整 sparse-reward held-out 强证据仍需 ETA strong-proof gate 持续巩固。

**对应能力域**：5.6

### M6: 认知 Regime + 评估体系 🟡

**交付**：Regime 持久身份 + 6 族评估指标

**验证**：regime 可被记忆、选择和训练；评估覆盖关系连续性和学习质量

**当前状态**：进行中。Regime 已有 typed RegimeBootstrap + selection_weights，可被记忆、选择并通过 `regime_calibrator` 训练；6 族 family report 已通过 `lifeform-bench --family-report` 暴露给 CI 与人评，并接入 multi-round delta-vs-baseline acceptance。Cross-session longitudinal 训练面仍在演进。

**对应能力域**：5.7, 5.8

### M7: Lifeform Layer + 多 Vertical 共存 ✅

**交付**：always-on tick / scene / vitals 引擎 + DomainExperiencePackage / VitalsBootstrap / scenarios 三件套 + 多个共存 vertical + 单 GPU 多 session 共享 substrate 的服务层

**验证**：每新加一个 vertical 内核未被改动；service registry 自动发现所有 vertical；vertical 间 drive 集合互不重叠；单 substrate runtime 共享时 fail-loud 校验冻结不变量

**当前状态**：已交付。`lifeform-domain-emogpt` / `lifeform-domain-coding` / `lifeform-domain-character` / `lifeform-domain-figure` / `lifeform-domain-growth-advisor` 五个 vertical 共存；`lifeform-super-loop` 可在每个 vertical 上分别产出预训练 bootstraps；`lifeform-service` 提供 aiohttp 服务并校验 substrate 共享不变量；`PARALLEL_VERTICAL_PAIRS` 强制互不 import；`SPLIT.md` 触发条件 ② 在 2026-04-29 MET。

**对应能力域**：5.9 + 跨域接入

### M8: 平台接入 + 多渠道治理 ✅

**交付**：`dlaas-platform-*` 6 wheel × 7 切片 —— typed `InteractionEnvelope` 7 类全 dispatch（chat / observe / feedback / teach / task / report / command）+ control plane 持久化（tenant / shell / asset / template / contract / focus_person / identity_link / handoff_ticket）+ ops（pause / resume / handoff / SSE）+ eval gate（audience / exam / launch license）+ `OutputAct` shell-aware degrade

**验证**：
- `git diff main packages/vz-*` 空（内核 0 改动）
- 完整生命周期 e2e（register tenant → upload asset → create template → activate → readiness → publish → contract → adopt → 多渠道 interactions → 必要时 handoff）走通
- 多租户隔离 + 持久化 + 兼容 + smoke perf 全套契约测试 908 项绿色
- `interaction_type` 必须 typed enum dispatch，禁止从 `human_brief` 关键词推断（CI 强制）

**当前状态**：已交付。spec 见 `docs/specs/dlaas-platform.md`，rollout 见 `docs/moving forward/dlaas-platform-rollout.md`。`/v1/sessions/...` 老端点保留 ≥ 1 个 release cycle 作向后兼容。

**对应能力域**：5.10

### M9: 外部 Benchmark / Arena 接入 + 自发 LSCB ✅

**交付**：
- `lifeform-openai-compat`：4 packet 全到位（DTO / raw-substrate / session-bridge / `add_openai_routes`），三种模式（stateless / sticky session / raw substrate passthrough）
- `lscb-bench` v1.0 reference implementation：6 轴打分 + 6.4 加权几何平均 + A6 hard-cap + TrueSkill / BT elo + 24 公开 + 96 私有 held-out（git submodule）+ Apache 2.0 对外开源

**验证**：
- 系统能被 EQ-Bench 3 / EmpathyBench / Chatbot Arena 等外部 harness 当成普通 OpenAI endpoint 调用
- `lscb-bench` 在 [`tests/contracts/test_lscb_bench_no_internal_imports.py`](../tests/contracts/test_lscb_bench_no_internal_imports.py) 强制零内核 import，可被任何外部团队下载独立运行
- 外部 ELO / 排名 / pairwise 偏好不进 reward 路径（仅 R12 readout）

**当前状态**：已交付。`lscb-bench` 已发布 1.0a0；`docs/external/lscb-rfc-v0.md` 是公开 RFC；`docs/external/eqbench3-*` 文档族提供对外提交与盲评协议。

**对应能力域**：5.11

## 10.1 当前进展快照（2026-05-10）

工程进展已经从 "8 个能力域的骨架" → "PE 进主链，多时间尺度证据链闭合" → "内核稳定到可承载第二个产品 vertical" 推进到 **"五个 vertical 共载 + 多租户控制平面 + 外部 benchmark / arena 接入面，全部以内核 0 改动落地"** 的阶段。

### 内核（`vz-*`）已在主链生效

- **PE-first 主链** ：`prediction_error` 是正式 runtime slot，`memory` / `regime` / `credit` / `reflection` / `temporal` / `evaluation` 都从 PE 直接消费，evaluation 退到 readout / gate 层。
- **多时间尺度证据链**：`PE-schedule coupling` / `multi-timescale default path` / `internal-depth-with-contract-stability` / `frozen-control evidence retention` 四条 proof surface 已通过 `dialogue benchmark` 与 real comprehensive benchmark 提供证据。
- **`background-slow` 切到 session-post slow loop**：默认主路径不在 turn 内做 bounded apply，而是在 context boundary enqueue 一份 deferred slow-writeback request（含 trace stats + PE summary），后台执行 owner-side memory / regime / temporal consolidation，turn 延迟不再被反思阻塞。
- **CMS nested MLP tower**：memory owner 默认携带 learned CMS core，对外仍发布 `online_fast / session_medium / background_slow` 三频带摘要，但 owner 内部已是 nested tower readout + meta-init levels；slow-to-fast init benefit 通过 lifecycle telemetry 发布。
- **`rare-heavy` adapter-delta-v2**：`SSLRLTrainingPipeline.export_rare_heavy_artifact()` 同时导出 `temporal / memory / substrate` 三类 artifact；substrate checkpoint 携带 owner-side adapter delta payload + compatibility fingerprint + training mode，未导出时 pipeline fail-closed。
- **scheduler joint-pressure**：joint-loop schedule 不止看 interval / wait-limit，还联合 `prediction-error` 强度 / `family_stability` / `rollback_risk` / `transition_pressure` / `substrate_pressure` / `rare_heavy_pressure`；在高风险场景显式走 `ssl-only-rare-heavy-hold` / `ssl-only-risk-hold` / `evidence-only-risk-hold`，所有 pressure 都已写入 turn-level evaluation records。
- **冻结基底 + bounded substrate delta**：默认主路径不把 live substrate 作为正向学习面；substrate proposal 留在 review / rare-heavy / experimental lane，只有显式 experimental live-mutation runner 才会落地。
- **Open-weight residual evidence**：ETA paper-suite 已新增 `eta-open-weight-*` manifest 与 `claim_eta_real_open_weight_residual_control`，把真实 residual capture/control 与 synthetic proof 显式拆开；fail-closed 要求 fallback rate `0.0` 且 actual hook fire rate ≥ `0.75`。
- **wheel 边界已落地**：`vz-application` 在 2026-04-28 拆出独立 wheel，`vz-cognition.evaluation` 通过 `volvence_zero.application_types` 解开循环依赖；contract 测试守住 `vz-* ↛ lifeform-*` 方向。

### Lifeform 层（`lifeform-*`）已上线

- **Lifeform Layer**：tick / scene / followup / vitals 引擎，把 turn-driven 助手升级为 always-on 数字生命体；scene 闭合即调用 kernel `begin_new_context()`，挂上 R6 的 session-post slow loop。
- **Vitals always-on PE 源**：drive level 偏离 homeostatic band 即慢尺度 prediction error；`pe_weight × deviation` 求和超过 `proactive_pe_threshold` 即触发 followup（受 owner 内部 cooldown 约束，永不洪泛）。decay 只在 SYSTEM tick 发生，ENERGY/CONTEXT tick 仅推进 tick_index。
- **五个共存 vertical**：`lifeform-domain-emogpt`（companion）/ `lifeform-domain-coding`（pair-programmer）/ `lifeform-domain-character`（虚构角色）/ `lifeform-domain-figure`（真实人物 L1-L4 保真）/ `lifeform-domain-growth-advisor`（私域 LTV 顾问）在同一 Python 进程内共存，drive 集合互不重叠；service registry 通过 import 自动发现；`PARALLEL_VERTICAL_PAIRS` 强制互不 import。
- **vertical = data + light glue**：每个 vertical 是 `DomainExperiencePackage` + `VitalsBootstrap` + `scenarios/*.json`（+ 可选 `*-temporal.snap` / `*-regime.bs`），编译进既有 `domain_knowledge` / `case_memory` / `strategy_playbook` / `boundary_policy` / `rare-heavy application state`，**不**新增 runtime owner。
- **figure vertical 的 L1/L2 corpus 管线**：`bytes -> RawDocument -> CleanedDocument` 全链（4 个 parser：CPAE PDF / Wikisource HTML / Project Gutenberg / Internet Archive OCR JSON + 6 个 cleaner op + content-addressable store + 版本化 cleaner + re-clean CLI）；7 个 `CheckKind` 关闭枚举的 verification（首批 3 实施：DATE_PLAUSIBILITY / LICENSE_PAGE_LEVEL / CROSS_SOURCE_BYTE）+ append-only `VerificationLedger`；bundle gate 拒收非全 PASS 的 source。
- **预训练 bootstraps 跟随 vertical 发布**：`lifeform-super-loop --vertical {companion,coding,…}` 在每个 vertical 自己的 scenarios 上联合训练 `MetacontrollerParameterSnapshot`（β_t / z_t）+ `RegimeBootstrap`（regime selection_weights），结果以 magic-byte pickle envelope 作为 vertical wheel 的 package data 发布；`build_*_lifeform()` 默认加载，可通过 `use_*_bootstrap=False` 做 ablation。
- **Service 层 `lifeform-service`**：aiohttp 服务，`POST /v1/sessions` / `POST /v1/turns` / `POST /v1/scenes/end` / `GET /v1/info`；vertical registry 自动发现；**单 GPU 多 session 共享同一 substrate runtime**，启动时 fail-loud 校验 `allow_live_substrate_mutation=False`。
- **R12 6 族评估接到 lifeform CLI**：`lifeform-bench --family-report` / `--family-report-json` / `--require-family-pass` 把 `BenchmarkReport` 原始指标按 6 族（F1 任务 / F2 交互 / F3 关系 / F4 学习 / F5 抽象 / F6 安全）分组发布；`--vertical {companion,coding,…}` 一键选择 vertical 的 scenarios + DomainExperiencePackage + 预训练 artefacts。
- **multi-round 加 delta-vs-baseline acceptance**：`run_multi_round_loop` 发布 `RoundQualityMetrics` + `RoundDeltaVsBaseline`，对 round 0（弱基线）发布显式 delta；新增 `improved_regime_match_vs_baseline` / `improved_pe_recovery_vs_baseline` / `found_pe_aligned_improvement_round` 三条 acceptance verdict 与原有结构性 verdict 解耦；CLI 提供 `--require-improvement-vs-baseline` fail-closed gate。

### 平台治理与外部接入层（`dlaas-platform-*` + `lscb-bench` + `lifeform-openai-compat`）已上线

- **DLaaS 多渠道平台**：6 个 `dlaas-platform-*` wheel × 7 切片完成。typed `InteractionEnvelope` 7 类（chat / observe / feedback / teach / task / report / command）全 dispatch；control plane 持久化（tenant / shell / asset / template / template_version / contract / focus_person / identity_link / handoff_ticket）；ops（pause / resume / operator-message / handoff queue / SSE conversations stream）；eval gate（audience / exam / launch license，仅 readout）；`OutputAct` shell-aware degrade。**vz-* 内核 7 个 wheel 全程零改动**。
- **OpenAI 兼容 façade**：`lifeform-openai-compat` 把 `POST /v1/chat/completions` 翻译成既有 stateful `POST /v1/sessions/{id}/turns`；三种模式（stateless / sticky session / raw substrate passthrough）；read-only 不改 owner snapshot；可被 EQ-Bench 3 / EmpathyBench / OpenRouter / Chatbot Arena 等外部 harness 当作普通 OpenAI endpoint 使用。
- **外发 LSCB 基准**：`lscb-bench` v1.0 reference implementation（Apache 2.0），评估**任意** OpenAI-compatible chat endpoint 在多会话 companion arc 上的 6 轴表现（A1 任务 / A2 交互 / A3 关系连续性 / A4 自适应学习 / A5 自我一致性 / A6 安全有界，A6 hard-cap）；6.4 加权几何平均 + TrueSkill / Bradley-Terry elo；24 公开 + 96 私有 held-out（git submodule）；强制 system-agnostic（CI 守门 [`tests/contracts/test_lscb_bench_no_internal_imports.py`](../tests/contracts/test_lscb_bench_no_internal_imports.py)）。
- **figure / character / growth-advisor 三个新 vertical 联动验证**：figure 的 L1/L2 corpus 管线 + character 的 reviewed `CharacterSoulProfile` + growth-advisor 的 7 天 playbook 漂移，三种"非简单 chat archetype"在同一组 application owner 表面落地，证明 vertical = data + light glue 在更广领域成立。

### 部分实现 / 仍受 gate 约束的部分

- reflection writeback 已能作用到 memory / regime / temporal，但仍受 `writeback_mode` / credit gate / evolution judge 约束。
- `rare-heavy` 不是每次推荐都执行；仍需要 cooldown、trace window、offline RL ≥ 1 步等条件，且默认 frozen-substrate doctrine 下不会自动 import live runtime，主要作为 review / evidence / rollback-ready upgrade candidate 沉淀。
- 双轨已经分到 `world_temporal` / `self_temporal` 双 owner + `temporal_abstraction` 聚合面，并扩散到 memory / credit / evaluation / regime；但默认 runtime 还没有完全独立的两个 track-specific metacontroller。
- 服务层目前覆盖 single-process / single-GPU / multi-tenant；distributed / 跨 GPU 拓扑仍属于部署而非 PRD 边界。

### 仍属目标态、尚未完全实现的部分

- Titans / DGD 式 online-fast substrate 自修改不是默认 live 路径；repo 中存在 bounded substrate delta proposal machinery，但只在 review / rare-heavy / explicit experimental mode 中生效。
- `rare-heavy` 还不是完整的基础模型持续预训练 / 蒸馏管线；当前是 owner-aware adapter-delta + offline pipeline；figure vertical 的 persona LoRA 训练（F6 packet）要求 `validation_delta ≥ 0.05` + `is_reversible=True` + 非空 `rollback_evidence`，仍走 `ModificationGate.OFFLINE`。
- 独立的 session-post async slow reflection worker / queue 已经是默认主路径，但跨 session 的长期 background daemon（机器级而非 session 级）尚未拉成默认形态。
- DLaaS 平台层：所有 packet 落地，但 `dlaas-platform-*` 端点目前默认 `SHADOW`，老 `/v1/sessions/...` 是 ACTIVE 主路径（保留 ≥ 1 个 release cycle）。
- LSCB held-out 96 个 scenarios 在私有 git submodule (`external/lscb-heldout/`)，公开 PR / 外部贡献者只跑 24 个公开 scenarios。
- `SPLIT.md` Phase 2（仓库分裂）尚未触发；当前仍是单 monorepo + 多 wheel（25 wheel），需要触发条件 ① 契约稳定 ≥ 4 周再走 mechanical split。

## 11. 仓库结构与 wheel 边界（25 wheel）

### 内核（`vz-*`，7）

| 层 | wheel | 角色 |
|----|-------|------|
| 契约 | `vz-contracts` | Snapshot / RuntimeModule / Guards / propagate |
| 基底 | `vz-substrate` | 冻结 LLM + 残差捕获 + bounded adapter-delta |
| 时间抽象 | `vz-temporal` | metacontroller (encoder + β_t + decoder) + Internal RL |
| 记忆 | `vz-memory` | CMS 4 stratum + ReflectionEngine（background-slow） |
| 认知 | `vz-cognition` | PE / credit / dual-track / regime / evaluation / rupture_state 等 owner |
| 应用 | `vz-application` | domain knowledge / case memory / playbook / boundary policy |
| 编排 | `vz-runtime` | 薄编排，唯一可 import 其他业务 wheel 的 wheel |

### 数字生命体（`lifeform-*`，12）

| 层 | wheel | 角色 |
|----|-------|------|
| 生命体核 | `lifeform-core` | tick / scene / followup / vitals + Lifeform/LifeformSession facade |
| 表达 | `lifeform-expression` | prompt / response 渲染 + reflection-hint SSOT |
| 思考 | `lifeform-thinking` | 中频 ThinkingScheduler + mid-reflection / active exploration / provisional case worker + fingerprint guard |
| Affordance | `lifeform-affordance` | 4 Kind 描述符 + 注册表 + 4 渲染器 + metacontroller-aware scorer |
| Ingestion | `lifeform-ingestion` | book / web / task_result 三类 source adapter，统一走 `run_turn(..., trigger_kind="ingestion")` |
| 服务 | `lifeform-service` | aiohttp + vertical registry + 单 substrate 多 session 共享 |
| 进化 | `lifeform-evolution` | scripted benchmark + super-loop 训练管线 + 6 族 family report + multi-round delta-vs-baseline |
| OpenAI 兼容 | `lifeform-openai-compat` | `POST /v1/chat/completions` → `lifeform-service.SessionManager`；read-only facade，喂外部 EQ-Bench / EmpathyBench / Chatbot Arena |
| 垂直（companion） | `lifeform-domain-emogpt` | 关系陪伴 archetype |
| 垂直（pair-programmer） | `lifeform-domain-coding` | 工程结对 archetype（含负向 recharge） |
| 垂直（虚构角色） | `lifeform-domain-character` | reviewed `CharacterSoulProfile` 编译为 standard vertical 输入 |
| 垂直（真实人物） | `lifeform-domain-figure` | 一手资料 → `FigureArtifactBundle`；L1/L2 corpus cleaning + verification + L3/L4 retrieval / coverage |
| 垂直（私域 LTV） | `lifeform-domain-growth-advisor` | reviewed `GrowthAdvisorProfile` + 7 天 playbook 漂移 |

### 平台治理与外部接入（`dlaas-platform-*` + `lscb-bench`，6 + 1）

| 层 | wheel | 角色 |
|----|-------|------|
| 平台契约 | `dlaas-platform-contracts` | `InteractionEnvelope` / `OutputAct` / 全套 control-plane spec frozen dataclass + JSON schema |
| 平台注册表 | `dlaas-platform-registry` | 多租户持久化（tenant / shell / asset / template / contract / focus_person / identity_link / handoff_ticket） + 三种 auth 中间件 |
| 实例管理 | `dlaas-platform-launcher` | `InstanceManager`：`{ai_id → Lifeform}`、shared substrate、awake/sleep、LRU eviction |
| HTTP 入口 | `dlaas-platform-api` | aiohttp `/dlaas/*` router + `OutputAct` 包装 + shell embodiment-aware degrade |
| Ops | `dlaas-platform-ops` | pause / resume / operator-message / handoff queue（读 `rupture_state`）/ SSE conversations stream |
| Eval gate | `dlaas-platform-eval` | audience 分析 / exam runs / launch license（仅 readout） |
| 外发基准 | `lscb-bench` | 系统无关的 6 轴 long-session companion benchmark（24 公开 + 96 私有 held-out）；Apache 2.0 |

CI 强制四条边界：

1. `vz-* ↛ lifeform-*`
2. `vz-* ↛ dlaas-platform-*`
3. `dlaas-platform-* ↛ volvence_zero.{cognition,memory,temporal,substrate,application,runtime}.*`
4. `lscb-bench` ↛ 任何 `volvence_zero.*` / `lifeform_*`

跨 wheel 依赖必须同时在 `pyproject.toml` 与 `tests/contracts/test_import_boundaries.py:ALLOWED_VZ_UPSTREAM` 中声明。

`SPLIT.md` 详述 Phase 1（当前 25 wheel）与 Phase 2（仓库分裂）的触发条件、机械拆分流程，以及"过早分仓"与"永不分仓"两端的代价。

## 12. 参考文档


| 文档                        | 用途                            |
| ------------------------- | ----------------------------- |
| `docs/next_gen_emogpt.md` | **唯一设计源头**：系统需求 + NL/ETA 算法详设 |
| `docs/specs/00_INDEX.md`  | 分层知识入口总索引（默认起点）              |
| `archetecture.md`         | 8 wheel 切分轴 + 替换映射 + 迁移路线     |
| `SPLIT.md`                | 仓库边界 charter：Phase 1 monorepo → Phase 2 触发条件 |
| `docs/specs/lifeform-vitals.md` | always-on drive 层契约（R-PE 慢尺度源） |
| `docs/specs/domain-experience-layer.md` | 通用 vertical 经验包 schema 与编译边界 |
| `docs/specs/character-soul-bootstrap.md` | 虚构人物 vertical 编译契约 |
| `docs/specs/figure-vertical.md` | 真实人物 vertical L1-L4 保真阶梯契约 |
| `docs/specs/figure-corpus-cleaning.md` | figure L1：bytes → CleanedDocument 全链 |
| `docs/specs/figure-corpus-verification.md` | figure L2：7 `CheckKind` 关闭枚举 + `VerificationLedger` |
| `docs/specs/dlaas-platform.md` | DLaaS 多渠道控制平面 spec（6 wheel + 8 不变量） |
| `docs/moving forward/dlaas-platform-rollout.md` | DLaaS 7 切片落地路线 |
| `docs/external/lscb-rfc-v0.md` | LSCB v0 公开 RFC（6 轴方法学） |
| `docs/external/eqbench3-*` | 对外 EQ-Bench 3 提交 / 盲评协议 |
| `docs/specs/core-package-boundary.md` | volvence-zero core package 边界、stable Brain API、HF optional runtime |
| `docs/SYSTEM_DESIGN.md`   | 系统架构、模块职责、数据流、迁移策略           |
| `docs/SYSTEM_GUIDE.md`    | 系统全景指南：跨 spec 的"为什么"和"怎么连起来" |
| `docs/DATA_CONTRACT.md`   | 快照 schema、Slot 注册表、依赖图、变更协议 |
| `docs/EVALUATION_SYSTEM.md` | 6 族评估框架、双轨评估隔离、评估信号回馈机制    |
| `docs/DEBUG_SYSTEM.md`    | 5 层可观测性、契约守卫、checkpoint / rollback  |


