# DINQ BP 对 Volvence 的启发

> Status: internal strategy note  
> Date: 2026-07-18  
> Scope: DINQ 四份材料与 Volvence 当前产品、商业化和证据体系的对照分析

## 1. 结论

DINQ 最值得学习的不是 “Human World Model” 这个词，而是它把以下五件事串成了一条商业闭环：

1. 用 Talent Mapping 这类高价值交付获得收入和客户；
2. 把交付过程沉淀为人才图谱与 person representation；
3. 通过 API / MCP 将能力嵌入招聘、销售、投资等 Agent；
4. 用 B 端查询需求推动 C 端用户认领自己的 profile；
5. 用认领后的 first-party 数据改善合法性、鲜度和模型质量。

对 Volvence 而言，相应动作不是复制一个 People API，而是：

> **先把“关系智能”做成客户愿意购买、可以直接行动、能够回收结果证据的产品，再让 DLaaS、MCP、人类世界模型和数字生命叙事成为放大器。**

当前最重要的问题不是继续扩充能力面，而是选择一个主商业楔子，形成：

```text
单一场景
  → 可行动的关系智能产物
  → typed outcome / 用户纠正
  → prediction error
  → scoped memory / controller / strategy 改进
  → 可验证业务结果
  → 复购与更多高质量数据
```

## 2. DINQ 做对了什么

### 2.1 先卖结果，再卖平台

DINQ 的 Talent Mapping 不是一个抽象的“人才模型”，而是可以直接交付的决策产物：

- 候选人分层；
- 创业倾向判断；
- 关键依据和反向约束；
- 状态触发信号；
- 监测频率；
- 下一步触达建议。

它把“找到一个人”继续推进到“现在是否值得行动、为什么、如何行动”。这使数据与模型在平台形成之前就能产生收入。

对 Volvence 的启发是：不能只展示“更长期、更懂人、更有记忆的对话”，而应输出客户可以据此行动的结果，例如：

- 关系阶段及其证据；
- 信任或流失风险；
- 尚未闭环的承诺和问题；
- 推荐的下一步关系动作；
- 应当停止商业推进或转人工的边界；
- 过去建议与真实结果之间的偏差。

### 2.2 把模型能力转译为业务节奏

DINQ 将候选人分为立即触达、定期观察、长期维护等层级，并为每层定义不同节奏。其价值不只来自分析，还来自资源配置。

Volvence 已经拥有 relationship state、regime、commitment、open loop、rupture/repair、prediction error 等内部状态。商业产品不应要求客户理解这些模块，而应由各 owner 在快照中发布足够描述，再由产品层转译为：

- 现在做什么；
- 为什么；
- 何时复核；
- 哪个证据会改变判断；
- 什么情况下应当退出、暂停或升级。

### 2.3 应用收入、专有数据和模型改进形成同一飞轮

DINQ 的叙事不是单独销售数据，也不是先训练完模型再找应用，而是让应用收入直接资助数据与 representation scaling。

Volvence 也需要把商业回路与学习回路对齐：

```text
真实长程互动
  → typed external outcome
  → prediction error
  → 主动选择高信息量轨迹
  → scoped relationship data
  → 记忆和控制器改进
  → 更高连续性、留存或业务转化
```

关键区别在于：这条飞轮必须通过 same-substrate ablation 和 evidence gate 验证。在 `first-stage-retained` 之前，不能对外宣称“人类世界模型已经成立”。

### 2.4 API / MCP 是分发层，不是价值本身

DINQ 把 People API 描述为 Agent economy 缺失的 human layer。这个包装有效，因为底层已有图谱、联系信息、人才异动和付费场景。

Volvence 已有 DLaaS、OpenAI-compatible facade、native typed envelope 和 MCP bridge。但协议数量本身不构成 traction。对外应强调：

- 接入后具体改善哪个长期结果；
- 哪些状态可以持续；
- 哪些变化可以审计；
- 用户如何纠正、删除或撤回；
- 何时自动停止、转人工或回滚。

### 2.5 Claimed Profile 同时解决增长、鲜度与合规

DINQ 的 Claimed Profile 让个人认领图谱节点，并控制：

- 暴露什么；
- 向谁暴露；
- 以何种条件接受触达。

它不仅是 C 端功能，也把 scraped data 转化为 consented first-party data，并提高数据鲜度。

Volvence 可以将这一机制转译为 **Relationship Passport**：

- 用户查看 AI 当前如何理解自己；
- 区分事实、推断、偏好、关系状态和共同记忆；
- 对错误信息进行纠正；
- 控制不同 AI、tenant、人物或场景可见的范围；
- 导出、删除或撤回；
- 对重要推断要求来源和更新时间。

Relationship Passport 不能只是一个档案页。它应成为 consent、scope、纠错、删除、owner hydration 和 provenance 的统一产品控制面。

## 3. Volvence 与 DINQ 的定位边界

两者都可能使用 “Human World Model” 叙事，但实际建模对象不同：

- **DINQ**：社会中的这个人是谁、做过什么、与谁连接、是否值得联系；
- **Volvence**：一个 adaptive agent 如何在与特定人的长期互动中预测、行动、纠错并形成关系。

因此不建议与 DINQ 抢同一个词位。Volvence 更适合占据：

- Relational Agency；
- Relationship Control Layer；
- Governed Adaptive Lifeform；
- 可持续、可治理的关系型 AI。

一句话建议：

> **把任意大模型的一次性 IQ，变成可以长期形成关系、持续纠错并受用户治理的 EQ + IQ。**

“Human World Model” 可以作为技术路线和验证命题，但不应在尚无 retain 证据时成为首要产品承诺。

## 4. 与 DINQ 的潜在互补

DINQ 类 People Graph 可以成为 Volvence 的外部环境数据源：

```text
DINQ People API / MCP
  → external evidence / affordance
  → canonical environment event
  → typed proposal
  → 对应 cognition owner 审核和写入
  → immutable snapshot
```

必须保持以下边界：

1. 外部 People API 不是 `user_model`、`relationship_state`、ToM 或 memory owner；
2. 不允许外部人物图谱直接写 cognition state；
3. 来源、时效、置信度、consent 状态必须可见；
4. 跨 tenant 不共享关系记忆或认知状态；
5. 与用户自身陈述冲突时，进入显式 reconciliation，而不是静默覆盖；
6. 人物事实可以作为环境证据，但不能被重新解释为已经建立的关系事实。

这意味着 DINQ 与 Volvence 更可能是基础设施互补，而非直接竞争：

- DINQ 提供外部世界中的人物与连接；
- Volvence 负责特定 AI 与特定人之间持续形成、治理和修复关系。

## 5. 不建议照搬的部分

### 5.1 不把定性研判包装成统计概率

人才地图已明确说明创业概率是定性研判，不是统计置信区间。Deck 中提到 person embedding、连接预测和人才流向评测，但未披露具体结果。

因此应学习其报告结构，不应直接学习概率表达方式。Volvence 的关系评分、风险预警和阶段判断应至少区分：

- qualitative judgment；
- model score；
- calibrated probability；
- externally validated outcome。

如果没有历史基准率、回测和校准证据，就不应展示为精确概率。

### 5.2 不同时铺开六个收入池

DINQ 将招聘、销售、投资、Agent infra、专家网络和信任验证放入同一叙事。对已有数据资产和客户的公司，这仍然可能显得过宽。

Volvence 当前同时拥有陪伴、数字员工、人物、私域顾问、DLaaS、Persona Market 和 Companion Bench 等方向，更需要主动收敛：

- 一个收入主路径；
- 一个证据路径；
- 一个长期平台 option。

其余方向应降为复用证明或未来选择权，不能都作为当前 GTM。

### 5.3 不把数据飞轮等同于跨用户学习

Volvence 的关系数据具有高隐私和强 scope 属性。不能为了模仿数据飞轮而建立未经同意的 automatic cross-user learning。

可行的飞轮应建立在：

- consented first-party data；
- 去身份化且经过审查的训练材料；
- per-tenant / per-user scope；
- 明确的数据撤回与删除；
- evidence-only aggregation；
- rare-heavy owner path。

## 6. 建议的 90 天行动

### 6.1 现在：冻结一个商业楔子

在 Figure-as-a-Service、Growth Advisor、Companion Bench 中只选一个收入主路径。选择标准：

- buyer 明确；
- 预算已经存在；
- 30 天内能观测结果；
- 需要长期关系，而不是一次问答；
- typed outcome 可以回流；
- 出错时有明确人工接管和责任边界。

其余两条分别作为证据资产或渠道，不同时作为主销售方向。

### 6.2 现在：制作可出售的 Relationship Intelligence Artifact

为选定场景设计一份客户可直接使用的关系智能产物，至少包含：

1. 当前关系阶段；
2. 关键证据和不确定性；
3. 用户目标与未闭环事项；
4. 承诺履行情况；
5. rupture / trust 风险；
6. 推荐动作与禁止动作；
7. 复核触发信号；
8. outcome 回填入口。

验收门槛不是“看起来聪明”，而是客户能据此采取行动并愿意复购。

### 6.3 现在：收窄对外叙事

首屏价值主张不应要求客户理解 NL、ETA、PE 或快照架构。

建议分三层：

- **产品层**：长期形成关系、持续纠错、可治理；
- **商业层**：改善留存、转化、复购、服务连续性或人工效率；
- **技术层**：多时间尺度学习、关系/任务双轨、prediction error、owner/snapshot 和可回滚控制。

### 6.4 随后：建立真实 outcome 数据回路

为选定场景定义：

- 业务结果；
- 关系结果；
- 用户纠正事件；
- 人工接管原因；
- 建议被采纳或拒绝；
- 延迟结果的回填窗口。

所有信号必须通过 typed intake 和 owner snapshot 进入系统，不允许从自然语言关键词重建。

### 6.5 随后：推出 Relationship Passport

先做最小版本：

- 查看；
- 纠正；
- 授权；
- 导出；
- 删除。

优先验证用户是否愿意纠正 AI 对自己的理解，以及这些纠正是否提升长期结果。不要先扩展为公开社交 profile。

### 6.6 观察：接入 DINQ 类 People Graph

可通过现有 MCP bridge / affordance 体系做 SHADOW 验证：

- 人物事实检索；
- 关系路径；
- 公开作品和共创证据；
- 联系方式或触达条件；
- 数据来源与更新时间。

只有在来源、scope、consent、冲突处理和审计全部明确后，才扩大接入范围。

## 7. 关键验证指标

商业验证优先于功能数量：

- 客户是否使用关系智能产物采取了实际行动；
- 30 天是否产生续费、复购或扩大试点；
- 相比同基底普通 LLM，是否提高长期任务完成率或关系连续性；
- 用户纠正 profile 后，后续 prediction error 是否下降；
- 建议被拒绝、转人工或触发边界的比例；
- 每项关系判断是否可以追溯到 owner snapshot 和外部证据；
- 删除、撤回和 tenant 隔离是否通过审计。

在 evidence gate 通过之前，以下表述应继续禁止：

- “人类世界模型已经证明”；
- “系统真正理解了人”；
- “关系状态预测准确率达到某个百分比”但无校准数据；
- “数据飞轮自动形成”但没有 consented outcome 回路；
- “跨用户学习提升模型”但没有 scope 和撤回机制。

## 8. 最终判断

DINQ 证明了“围绕人”的模型可以从高价值情报交付切入，而不必先等待一个完整平台成熟。

Volvence 应采用同样的商业方法，但保持不同的产品核心：

> **DINQ 把人变成机器可查询的对象；Volvence 让机器在与人的长期关系中持续预测、行动、纠错，并接受人的治理。**

我们的近期目标不是再增加一组技术能力，而是把已经存在的关系状态、记忆、PE、regime、commitment 和治理机制收束成一个客户愿意持续购买的结果。

## 9. 材料来源

外部材料：

- `预测性人才地图_AI研究员创业倾向研判_2026H2.pdf`
- `DINQ Investor OnePager(1).pdf`
- `DINQ_Roadmap_Deck_202607(1).pdf`
- `DINQ_Roadmap_Deck_EN(1).pdf`

内部对照：

- `docs/prd.md`
- `docs/next_gen_emogpt.md`
- `docs/business/commercialization-assessment.md`
- `docs/business/business-product-roadmap-v2-cn.md`
- `docs/specs/dlaas-api-v1.md`
- `docs/specs/dlaas-platform.md`
- `docs/specs/mcp-bridge.md`
- `docs/specs/human-world-model-ablation.md`
- `docs/specs/evidence_program.md`
- `research/strategy/human-world-model-thesis-2026-06.md`
