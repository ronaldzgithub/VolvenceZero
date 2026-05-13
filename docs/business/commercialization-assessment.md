# VolvenceZero 商业化评估

> Status: draft v0.1
> Last updated: 2026-05-13
> 适用范围：本文件是 VolvenceZero（VZ）从工程态走向产品态/商业态的**商业评估底稿**，不是营销稿，不是路演稿，不是 PRD。
> 与其他文档的关系：
> - `docs/prd.md` — 产品需求文档（系统目标态、能力域、里程碑）
> - `archetecture.md` — 架构边界（25 wheel × 3 层）
> - `docs/moving forward/summary.md` — 团队对自身工程信心的冷静校准（项目自评）
> - `docs/closed-alpha-api-service.md` — 已上线的 closed-alpha 服务面
> - 本文件 — 把上面这些**已交付的工程能力**翻译成**可执行的商业路径**，并对每条路径给出现实的概率/成本/护城河判断
>
> 写作原则：
> 1. 不夸大已有能力；以仓库现状为锚点（截至 2026-05-13）
> 2. 不假装"商业问题就是技术问题再 +1 层"——商业判断需要独立的 GTM、单位经济、市场结构论证
> 3. 对每条路径都给出**kill criteria**（什么情况下应该砍掉这条路径），而不是只给 OKR
> 4. 与 `docs/moving forward/summary.md` 的 (a)/(b)/(c) 三档目标信心标定保持一致：本文件主要服务于目标 (b)（"可商业化数字生命"，25-40% 概率），不为目标 (c)（强义 AGI）写商业路径
> 5. **不**做"凡是 AI 公司都该做"的通用判断（数据飞轮、生态、平台）；只评 VZ 已经持有或快持有的差异化资产能撑起什么

---

## 目录

1. 系统的商业本质：我们卖的是什么
2. 真正的差异化资产盘点（What's the moat）
3. 市场结构与对手坐标（Who else is here）
4. 商业化路径备选清单（6 条 + 各自的概率/成本/护城河）
5. 推荐路线：12-24 个月的产品化排序
6. 单位经济初稿（Unit Economics）
7. 进入策略（GTM）
8. 关键风险与 Kill Criteria
9. 12-18 个月可写 OKR 草案
10. 不应该做的事（Anti-goals）
11. 复盘机制：每 90 天回看本文件应该改什么

---

## 1. 系统的商业本质：我们卖的是什么

### 1.1 一句话总结

VZ 不是一个"更聪明的 LLM 包装"，也不是一个"AI 陪聊 App"。它是**一台被工程契约约束的"持续关系型代理"运行时**——具备 4 个时间尺度的学习循环、连续记忆、双轨（任务 / 关系）分离、可审计内部状态、可回滚自修改门控、多 vertical 共载、多租户控制平面、shell-aware 输出降级。

把它翻译成商业语言：

- 我们能让一个 AI 持有 user-specific 的关系状态**跨会话连续**，并且这种连续性可被审计、可被用户删除、可被合规观察。
- 我们能让一个 AI **拒答**它没被授权说的话（figure 垂直 L4 / growth-advisor 的 boundary policy）；这件事 LLM 厂商默认做不到，因为他们的 instruction following 不是基于结构化 owner，而是 prompt-level 影响。
- 我们能让一个 AI **同时承载多个 vertical**（5 个并列），共享同一个冻结 substrate，但 drive / regime / boundary 互不污染。
- 我们能让 B 端运营方在不改内核的情况下，按 tenant / shell / asset / template / contract 全套资源配置自己的"数字生命"实例。

商业上 VZ 卖的不是"模型"，而是 **"关系连续性 + 可治理性 + 多角色复用"** 这一组特性的组合包。这三件事任何一件都不能直接在 OpenAI / Anthropic 的 API 上拼出来。

### 1.2 我们没有在卖什么

为了避免战略漂移，明确列出 VZ 不应该宣称在卖的东西：

| 容易被误用的卖点 | 为什么不应该卖 |
|---|---|
| "比 GPT/Claude 更聪明" | substrate ceiling 锁死，IQ 路径不是 VZ 的护城河（见 `summary.md` §1） |
| "AGI 路径" | 架构是 cognitive AGI 的**容器**而非**实现**，把容器当成实现卖会被工程现实打脸 |
| "通用记忆插件" | OpenAI Memory / Anthropic Projects / Mem0 / Letta 已在通用 RAG 记忆赛道；拼通用记忆是输的 |
| "Agent 框架" | LangGraph / AutoGen / Crew 已占住编排层；VZ 的 contract runtime 是给**自己**用的，不是给开发者用的通用框架 |
| "情绪识别 / 共情打分" | 这些是 evaluation readout，不是产品；卖打分会让客户期望管理崩盘 |
| "AI 心理咨询师" | 牌照 / 责任 / 合规直接踩雷，不在当前 alpha 服务安全边界内（见 closed-alpha-api-service §Safety Boundary Minimum） |

### 1.3 三种可能的"产品形态"分类

把后续路径前置到三种形态上，便于第 4 章对路径分类：

- **形态 A — 终端产品（B2C / B2C2C）**：用户面向我们或我们面向 KOL/IP 出版方的终端 App / Web / 小程序，付订阅或单次。
- **形态 B — 可托管的 AI 实例（DLaaS, B2B）**：客户接入 DLaaS 平台，开自己的 ai_id × shell × template，按调用 / 按时长 / 按席位计费。VZ 是底层运行时 + 治理面。
- **形态 C — 行业可信度 / 标准化资产（Trust capital）**：开源的 `companion-bench`、对外可读的 R12 6 族评估、可审计 evidence bundle。这个本身**不直接收钱**，但抬高 A、B 两种形态的可信溢价。

每条商业路径都至少属于上述一种形态，第 4 章会标注。

---

## 2. 真正的差异化资产盘点（What's the moat）

护城河的判定标准只有一条：**对手要在 12 个月内复刻这件事，需要付出的工程量是否显著大于 6 个月**。下面按"现成的、6 个月内可包装、12 个月才能兑现"分三档。

### 2.1 已经现成、可以马上拿出去用的（Tier-1）

这些东西仓库里已经写好、有契约测试守门、可以现场 demo：

| 资产 | 来源 | 商业价值 |
|---|---|---|
| **Rupture/Repair 闭环 + typed dialogue outcome** | `vz-cognition.rupture_state` + `lifeform-service` 的 `/v1/sessions/{id}/dialogue-outcomes` typed enum | 用户显式说"你在 over-directive"→ 系统改下一轮表达 → 出 `rupture_repair` durable memory；这是单 LLM API 在产品层做不出的"我听见你在说我哪里错了" |
| **Scoped memory + 删除路径** | `volvence_zero.memory.UserIdentity.scope_key` + `DELETE /v1/users/me/memory` + deletion evidence | 任何关心"用户记忆隔离 + 用户可被遗忘"的 B 端客户（教育、金融、医疗、政务）需要这个，OpenAI / Anthropic 默认不提供 |
| **5 个并列 vertical 在同一进程共载** | `lifeform-domain-{emogpt,coding,character,figure,growth-advisor}` + `PARALLEL_VERTICAL_PAIRS` CI 强制 | "一个内核服务多个垂直角色"是 DLaaS 价值主张的实证；没有这个，多租户控制面是 PPT |
| **typed `InteractionEnvelope` 7 类** | `dlaas-platform-contracts` + `dlaas-platform-api` | 客户端不能用"自然语言夹带"搞出 rupture / handoff，必须显式 typed 调用——这是 enterprise 合规审计的硬需求 |
| **handoff queue + pause + SSE 观察** | `dlaas-platform-ops` | 真人客服接管 / 暂停账户 / 实时监控对话流——B 端 ops 团队的最低要求 |
| **OpenAI 兼容 facade（read-only）** | `lifeform-openai-compat` 4 packet 全到位 | 任何外部 Arena / benchmark / 客户的现有 OpenAI SDK 客户端可以**零改造**接入 VZ 实例；同时 read-only 守住"外部 harness 不污染 kernel" |
| **`companion-bench` v1.0 (Apache 2.0)** | `packages/companion-bench` 24 公开 + 96 私有 held-out | **行业可信度资产**：这是 VZ 把自己定义为"长程陪伴评估标准制定者"的入场券，对手要复刻这套（A1-A6 6 轴 + TrueSkill + held-out submodule）至少 3-6 个月 |
| **launch license gate + audience 分析** | `dlaas-platform-eval` | 客户上线一个 ai_id 之前必须跑 exam / audience 才放行——这是给 B 端"能不能上线一个 persona"的工程化答复 |

**Tier-1 总结**：这些不是 PR 里说的"某天会有"，而是**已经在 main 分支的代码 + 契约测试**。任何严肃 B 端尽调，把上面这套打包演示一次，是可以拿到 PoC 合同的。

### 2.2 6 个月内可被严肃包装成产品的（Tier-2）

骨架在，缺的是"客户实际买得到、用得起、能上量"的产品化外层：

| 资产 | 当前状态 | 6 个月内的成熟度需要做什么 |
|---|---|---|
| **Figure vertical L1+L3+L4 零 GPU 上线** | `lifeform-domain-figure` Wave A-G 已落，L1/L2 corpus 管线已通；L2 / 加强 L1 走 `ModificationGate.OFFLINE` | 选 1-2 个公共领域人物（爱因斯坦 / 居里夫人 / 苏轼 / 鲁迅）做完整 corpus → bundle → adopt → activate 全链 demo；做到端到端可点击体验 |
| **Growth-advisor 7 天 playbook + 4 反销售边界** | `cheng-laoshi` profile 已编码（5 archetype × 7 day × 4 funnel × 4 boundary），`bp-no-hard-sell/overclaim/flooding/judgmental` 守门 | 接 1 个真实私域运营客户跑 30 天试点；把 audience report 做成"运营月报"格式 |
| **Coding vertical guided_exploration regime + 负向 recharge** | `lifeform-domain-coding` 的 `direction_certainty` drive，`guided_exploration` regime 下用负向 recharge | 接到 IDE / Cursor / VSCode 插件层，实测"模糊需求澄清 / 架构方向探索"是否真带来更高接受率 |
| **Closed alpha 服务的可付费化** | `lifeform-serve --alpha-enabled` 已跑通，has user allowlist + memory deletion + weekly report | 加：付款接入 / 多用户隔离的 ops dashboard / 客户级评估报表 |
| **Companion-bench 公开榜单 + 第三方提交流** | spec 完整，`heldout_loader` 私有 submodule | 公开域名 + 提交 RFC + 主流闭源/开源模型先打一遍榜（含 GPT-5 / Claude Opus 4.7 / DeepSeek / Qwen / Llama 等）；榜单本身是**冷启动品牌资产** |

### 2.3 12-18 个月才能真兑现的、不要现在拿去卖（Tier-3）

这些是 `summary.md` 里明确列为"未经验证"或"机制级假设"的，**不要**现在打包成产品故事：

- **Latent action RL on frozen base 在开放对话上的有效性**——CoLA/FR-Ponder 只在受限推理任务证过；卖之前需要 SYS-2/SYS-5/COG-7 evidence。
- **PE-as-primary-signal 在开放对话上的效率**——`prediction_error` 是 owner，但开放域的 PE ground truth 还没操作化。
- **Persona Vectors / SAE 作为身份漂移监控**——COG-3/COG-8 reproducibility 风险大。
- **多代际胜率证明在朝 cognitive AGI 收敛**——EVO-6 只能证"新一代比老一代好"，不能证"在朝目标收敛"。
- **跨用户的关系学习 / 群体级记忆**——目前 closed alpha 显式禁止 automatic cross-user learning。

把 Tier-3 的东西作为"将来可能"放在愿景章节没问题；放进合同 SLA 会出事。

### 2.4 资产到护城河的链接逻辑

资产本身不构成护城河，资产 × **客户切换成本**才构成护城河。VZ 资产对应的切换成本来源：

1. **关系记忆切换成本**：用户在 VZ 实例上积累的 scoped durable memory + rupture_repair history 越多，迁出成本越高（因为对手 LLM API 的 memory 不可移植回这种 schema）。
2. **审计 evidence 切换成本**：B 端客户接入 VZ 后产生的 `evidence_root_dir/sessions/...json` 是合规材料；换平台等于丢合规历史。
3. **vertical 编译资产切换成本**：figure 的 `FigureArtifactBundle`（含 retrieval index / coverage map / style prior / 可选 steering / persona LoRA）是 VZ 编译产物，不是通用 LoRA；换底座要重新编译。
4. **`companion-bench` 排名切换成本**：如果 VZ 主导这条 benchmark 并被引用，对手为了对得上口径就得自己接 OpenAI 兼容 facade 接入跑分——这个生态位被早占住值钱。

护城河 = 上面 4 条切换成本的复利。任何一条单独不够强，但叠在一起，会产生**对手要打 VZ 至少要在 4 个面同时打**的局面。

---

## 3. 市场结构与对手坐标（Who else is here）

把 VZ 放进真实的赛道地图里看。每个赛道里 VZ 的位置不同，能不能赢取决于**这个赛道的胜负手是不是恰好踩在 VZ 的护城河上**。

### 3.1 赛道一览（截至 2026-Q2）

| 赛道 | 主要玩家 | 胜负手 | VZ 是否踩在胜负手上 |
|---|---|---|---|
| **通用 AI 助手** | OpenAI / Anthropic / Google / DeepSeek | 模型能力 + 分发渠道 + 价格 | ❌ 不踩。VZ substrate 是冻结的开源模型，IQ 上无优势 |
| **AI 陪伴 / 情感陪聊（C 端）** | Character.ai / Replika / Janitor / 星野 / Talkie / 筑梦岛 | 角色 IP 库 + UGC 生态 + LTV 单价 + 留存 | ⚠️ 半踩。VZ 的关系连续性 + 修复闭环是差异化，但**冷启动 IP 库 + UGC 生态**完全没有 |
| **真人/历史人物数字复生** | HereAfter AI / Storyfile / DeepBrain AI / 国内多家"数字孪生"公司 | 一手语料合规 + 家属/IP 授权 + 表演（视频/语音）层 | ✅ 部分踩。VZ 的 L1-L4 保真阶梯 + 引证拒答 + 不可变 bundle 是差异化；但 VZ 没有视频/语音表演层 |
| **私域 LTV 顾问 / 教育-母婴-健康** | 各行业的人工社群运营 + ChatBot SaaS（如腾讯企微 + 第三方 SCRM） | 行业 know-how + 运营模板 + 低单价高流量 | ✅ 踩。VZ 的 7 天 playbook + 反销售边界 + audience 分析是行业 SaaS 不会做的"治理面" |
| **AI 编程助手** | Cursor / GitHub Copilot / Cline / Aider / Continue / 通义灵码 | IDE 集成 + 模型能力 + 上下文工程 | ❌ 不踩。Cursor 已经是赛道事实标准；VZ 的 guided_exploration regime 是有趣但**不构成换工具的足够理由** |
| **B2B Agent 平台 / Workflow** | LangChain / LangSmith / CrewAI / AutoGen / Dify / Coze / 字节扣子 | 编排灵活性 + 集成生态 + 可观察性 | ❌ 不踩。VZ 的 contract runtime 是给自己用的，不是给开发者搭流程的 |
| **合规级 AI / 受监管行业** | Salesforce Einstein / Microsoft Copilot for Finance / 各行业垂直 SaaS | 合规背书 + 行业渠道 + 已有客户基础 | ⚠️ 半踩。VZ 的 audit + scoped memory + handoff queue 是合规级别的，但**没有合规背书 + 没有渠道** |
| **数字员工 / Outbound 销售自动化** | Reflex / 11x.ai / 国内多家"AI 销售"创业 | ROI 可验证（替人成本）+ 转化率 | ⚠️ 半踩。VZ 的 4 反销售边界（no-hard-sell 等）方向**反直觉**——故意不卖力卖；这是治理价值，但需要找到买"治理"而不是买"转化"的客户 |
| **AI 评估 / Benchmark / Arena** | Chatbot Arena / EQ-Bench / OpenCompass / HumanEval 系 / Stanford HELM | 学术认可 + 提交量 + 中立性 | ✅ 踩。`companion-bench` 在长程陪伴 niche 上**没有强对手**；EQ-Bench 3 在情绪轴有优势但没有跨会话 arc |

### 3.2 关键观察：哪些赛道里 VZ 不该去

- **通用 AI 助手 / IDE 编程助手**：直接放弃。VZ 不是 substrate 玩家，去打 substrate 玩家就是送死。
- **B2B Agent 平台**：放弃。Dify / Coze / LangChain 已占住"低门槛搭流程"生态位；VZ 的 contract runtime 哲学（先立法、再立法解释、最后写代码）和"人人可搭流程"是**反向的**，强行进入会和自己的工程纪律打架。
- **AI 数字员工 outbound 销售**：风险大。VZ 的 boundary policy / no-hard-sell / no-overclaim 在"销售转化"赛道是**负产品特性**——客户雇 AI 销售就是要它推销，不是要它克制。除非 pivot 出一条"高敏感行业的合规销售"细分（医疗、保险、法律咨询），否则避开。

### 3.3 关键观察：哪些赛道里 VZ 有不可复制的位置

- **真人/历史人物数字复生（B2C2C）**：DeepBrain / Storyfile 强在"视频/语音外壳"，弱在"事实保真 + 引证 + 拒答"。VZ 的 L3/L4 让 figure vertical 在**学术、博物馆、教育、出版**这种**事实正确性 > 表演逼真性**的客户身上有差异化。
- **私域 LTV 顾问（B2B）**：现成 SCRM 工具卖给运营总监，强在 CRM/标签/触达；VZ 卖给品牌方/合规总监，强在"AI 不会乱推销 / 用户可删 / 治理面 + 7 天 playbook + 月度运营报表"。这是**定位差**而非**功能差**——同样的客户群体不同的购买动机。
- **长程陪伴评估标准（行业可信度）**：`companion-bench` 在跨会话陪伴评估上**没有占住生态位的对手**。先把 GPT-5 / Claude Opus 4.7 / DeepSeek / Qwen / Llama 在 24 公开 + 96 私有 held-out 上跑出第一份"chatbot 长程陪伴排行榜"，谁来都得用我们的口径解释自己。

### 3.4 时间窗口的现实判断

VZ 的差异化窗口能开多久，取决于 OpenAI / Anthropic / Google 把这些能力补齐的速度：

| 能力 | 大厂当前进度 | 大厂补齐时间窗预估 |
|---|---|---|
| 跨会话 user memory | OpenAI Memory（已上）/ Anthropic Projects（已上）/ Gemini （已上） | **已基本补齐**——不是 VZ 的差异化点 |
| 用户可删除记忆 | OpenAI 已支持 | **已补齐**——但是粗粒度 |
| typed feedback enum | 大厂普遍是 thumbs up/down + free text | 12-18 个月不会做，因为**和大厂"通用助手"叙事相悖** |
| Persona artifact 编译（含拒答） | 大厂偏好 prompt-level persona | 12-24 个月不会做，因为**和大厂"模型一致性"哲学相悖** |
| 跨会话长程陪伴评估口径 | 大厂没有动力 | 24 个月+不会做，**niche 太小** |
| 多 vertical 同进程共载 | 大厂走 GPT Store 多实例路径 | 18 个月+不会做，因为**架构假设不同** |
| 治理级 audit / handoff / pause | 大厂偏好"信任厂商" | 24 个月+不会做，**和大厂垄断叙事相悖** |

**结论**：VZ 的差异化点中，至少 4 项（typed feedback / persona 拒答 / 长程评估 / 治理级 audit）**12-24 个月不会被大厂主动补齐**——不是因为大厂做不到，而是因为这些做了反而和大厂"通用 + 信任 + 一致"的叙事冲突。这个窗口就是 VZ 的商业窗口。

但需要警惕：**Anthropic 走 Constitutional AI / safety 路线**比 OpenAI 更可能切入"治理级"叙事；如果 Anthropic 在 12-18 个月内推出 "audit-grade chat completions"，VZ 的 enterprise 治理路径会被部分挤压。这是"差异化窗口可能比预期短"的最大风险。

---

## 4. 商业化路径备选清单

每条路径给出：形态分类、客户、价值主张、定价模型、所需投入、12 个月内可达成的"可验证里程碑"、概率估计、kill criteria。

> **概率说明**：每条路径的"12 个月内做出第一笔签约/上线/付费收入"的概率，区间在 5%-60%。这些数字不是希望值，是**给团队做资源分配排序用的**。

### 4.1 路径 P1：Figure-as-a-Service（真实人物数字复生）— 概率 35-50%

- **形态**：A（B2C2C，IP 出版方/博物馆/教育机构 → 终端用户）+ B（B2B 实例）
- **客户**：
  - **第一波**：博物馆 / 大学 / 公共图书馆（爱因斯坦、达芬奇、苏轼、鲁迅、特斯拉）—— 公共领域，没 IP 风险
  - **第二波**：作家遗产管理会、学术机构（已逝近现代学者，需家属/版权方授权）
  - **第三波**：在世人物自授权（KOL / 学者 / 企业家做"我的数字分身")
- **价值主张**：
  - "**它说的每一句话都能溯源**"——L3 引证 + L4 拒答让博物馆/教育机构的法务可以签字
  - "**人物的早期 vs 晚期立场可以分开演**"——TimeWindowedView 是 character vertical 没有的能力
  - "**bundle 是不可变快照，可重现**"——`figure_artifact_id` 就是审计 ID
- **定价模型**：
  - 一次性 **Bundle 编译费**（10-50 万人民币 / 个，取决于 corpus 规模 & 是否需要 OFFLINE-gate L2 steering / persona LoRA）
  - **托管费**（按月，按调用量阶梯，1-10 万 / 月）
  - **API 透传费**（substrate 调用按 token，原价 +20-30% 加成）
  - 如果客户走 B2C2C：**对终端用户分账**（比如 5 元 / 次对话，B 方分 50%）
- **所需投入（6-12 个月）**：
  1. 选 1-2 个公共领域人物，自费做完整 corpus → bundle → adopt → activate 全链 demo（爱因斯坦最优，corpus 充足且公共领域）
  2. 视频/语音外壳通过合作方接入（不自研）
  3. 1-2 个种子博物馆/大学客户做共建合作
  4. 案例 + 学术论文（共建客户出署名）= 行业可信度抬升
- **12 个月里程碑**：
  - 至少 1 个公开人物 demo 跑通端到端
  - 至少 1 个签约客户付编译费 + 3 个月托管
  - 累计签约金额 > 100 万人民币
- **概率拆解**：35-50%（demo 可控；客户签约取决于学术/博物馆采购周期，慢且贵）
- **Kill criteria**：
  - 12 个月没有任何博物馆/大学/出版机构愿意付编译费 → 砍
  - 头部数字复生公司（DeepBrain / Storyfile）在 12 个月内推出"引证保真 + 拒答"功能 → 重新评估护城河
  - L2 steering / persona LoRA 真实开放权重底座（Llama / Qwen）效果在 sample 测试中无显著差异（< 0.05 validation_delta）→ Bundle 编译价值被打折，但 L1+L3+L4 路径仍可独立卖

### 4.2 路径 P2：Private-Domain Growth-Advisor（私域 LTV 顾问）— 概率 30-45%

- **形态**：B（B2B DLaaS）
- **客户**：
  - 母婴 / 早教 / 儿童营养品（profile 已编码 `cheng-laoshi`）
  - 成人教育 / 职业培训 / 留学顾问
  - 财富管理 / 保险代理（在合规允许的咨询范围内）
  - 心理咨询机构（**仅作转介前的 triage**，不做诊断）
- **价值主张**：
  - "**AI 顾问不会硬推销**"——`bp-no-hard-sell` 是合同条款不是 prompt
  - "**用户 7 天养成阶段是工程化的**"——`growth_advisor:day1...day7` 通过 `applicability_scope` 真路由，不是关键词匹配
  - "**月度可审计运营报告**"——`audience` 分析 + `dialogue_external_outcome` typed enum 自动化产出
  - "**合规：用户可删 / 处置可查 / 转人工可控**"——handoff queue + 删除路径
- **定价模型**：
  - **席位制**（每个被托管的私域社群 / 顾问账号 = 1 席位，2000-8000 元/月/席位）
  - **调用量阶梯**（substrate token 透传 + 加成）
  - **L2 编译选配**（如果客户提供自己 reviewed 顾问 profile，按 figure 编译路径加成 20-50 万一次性）
- **所需投入（6-12 个月）**：
  1. `cheng-laoshi` profile 接 1 个真实的母婴/早教私域客户 30 天试点
  2. 把 weekly-report 升级成 **"运营总监能看懂的月报"**（rupture 数 / repair 率 / boundary 触发数 / 用户活跃度）
  3. 接付款 + 多客户隔离 + ops dashboard
  4. 行业模板复用：基于 `cheng-laoshi` 抽出更通用的 `GrowthAdvisorProfile` 编译协议，让客户填表→自动 compile
- **12 个月里程碑**：
  - 3-5 个签约付费客户（席位制）
  - 至少 1 个客户跑满 6 个月并续签
  - 累计 ARR > 200 万人民币
- **概率拆解**：30-45%（客户决策链短于路径 P1，但市场上 SCRM/SaaS 已经卷得很惨；VZ 的差异化要客户**懂治理价值**才能买单）
- **Kill criteria**：
  - 6 个月内没有任何客户愿意付席位费 → 砍或转 P3
  - 试点客户 30 天数据显示 boundary policy 触发率 < 5% 或 > 50%（要么没起作用，要么过度限制） → 调架构再试一次
  - 客户拿到月报后的反馈是"我看不懂这些数据" → 报表设计错了，先 fix 报表

### 4.3 路径 P3：Companion / Emogpt 直营 C 端（关系陪伴）— 概率 15-25%

- **形态**：A（B2C 订阅）
- **客户**：付费的成年陪伴用户（明确**排除未成年人**，已写入 closed-alpha non-goals）
- **价值主张**：
  - "**它真的记得你的名字、你说过的话、你不喜欢什么**"——scoped durable memory + relationship-summary
  - "**它说错话的时候，你说一句它就改**"——typed `OVER_DIRECTIVE` / `MISSED` 反馈进 rupture loop
  - "**不暧昧、不诱导、不卖催化氪金**"——boundary policy 一开始就内置（差异化于 Replika/Talkie 的氪金机制）
- **定价模型**：年订阅 600-1200 元 / 用户（对标 Replika Pro 80 USD/年 + 中国市场议价）
- **所需投入（12-18 个月）**：
  1. C 端 App / 小程序前端（**这块投入巨大且非 VZ 强项**）
  2. 用户增长 / 投放（**烧钱赛道**）
  3. 内容审核 + 法务（成年用户身份核验 / 违禁话题阻断）
  4. 24×7 运维（手机端 push / followup / tick）
- **12 个月里程碑**：
  - DAU > 1 万 / MAU > 5 万
  - 付费转化 > 3% / 留存 D30 > 25%
  - ARR > 500 万人民币
- **概率拆解**：15-25%——C 端陪伴是**高烧钱、高竞争、低差异化感知**的赛道。VZ 的关系连续性在产品端可感知度需要长期使用才显现（>30 天），而 D30 留存赛道惨烈
- **Kill criteria**：
  - 12 个月 DAU < 3000 → 砍
  - CAC > 6 个月用户 LTV → 砍（陪伴赛道单价低，CAC 一旦失控就回不来）
  - 出现一次严重内容安全事故（青少年 / 自伤 / 政治）→ 立即停服反思，不要硬撑
- **风险特殊提示**：这条路径的风险**最高且不只是商业风险**。陪伴产品的伦理 / 法律 / 公关风险高于其它路径，团队应慎重评估"是否要做 C 端这件事"本身

### 4.4 路径 P4：DLaaS B2B 通用平台（让别人在 VZ 上养自己的数字员工）— 概率 25-40%

- **形态**：B（B2B 平台）+ C（治理标准化）
- **客户**：
  - 大型企业内部数字员工建设（HR onboarding 助手 / 法务 FAQ / 客服 triage / 合规咨询）
  - 政府 / 国企的"政民交互"机器人（这条线在中国是**有强需求但合规门槛高**）
  - 大学 / 教育机构的"学习陪伴助理"（差异化于通用 chatGPT 接入：跨学期连续 + 可删 + 治理）
  - 2B SaaS 厂商集成（嵌入对方产品，OEM 形态）
- **价值主张**：
  - "**你的所有 ai_id 都在你的私有部署 VZ 实例里跑，多个 vertical / 多个 persona / 多个 shell embodiment 共享一个 substrate**"
  - "**typed audit + handoff + pause + delete = 满足合规审计**"
  - "**OpenAI 兼容 facade = 你现在的客户端代码不用改**"
- **定价模型**：
  - **基础授权费**（私有部署版，30-200 万 / 年，按 ai_id 数量分档）
  - **调用量阶梯**（substrate 自部署 / 客户提供 substrate / VZ 代部署 三档）
  - **专业服务**（vertical 编译 / 模板定制 / 集成开发，T&M 计费）
- **所需投入（12-18 个月）**：
  1. 把 alpha 服务升到 production（多租户隔离测试 / SLO / 灾备 / 监控）
  2. 私有部署版 packaging（Docker / K8s helm chart / 离线 license server）
  3. 销售/解决方案团队（**这是 VZ 团队当前结构里没有的能力**）
  4. 1-2 个**有名字的灯塔客户**作为参考案例
- **12 个月里程碑**：
  - 1 个签约灯塔客户（不要钱也行，要 case study）
  - 1-2 个付费客户
  - 累计 ARR > 300 万人民币
- **概率拆解**：25-40%（B2B 周期长，但客单价高；最大不确定性是"团队是否有能力跑 enterprise 销售"——这是 VZ 团队当前没有验证过的能力）
- **Kill criteria**：
  - 12 个月没有任何灯塔客户愿意做 PoC → 砍
  - 灯塔客户 PoC 后**不愿付费**（说"功能 OK 但没 ROI"） → 价值主张错位，需要重新调整定位
  - Anthropic / 国内大厂在 12-18 个月内推出"audit-grade chat completions" → VZ 这条路径需要**重新打差异化牌**，护城河收缩

### 4.5 路径 P5：Companion Bench 行业标准化（行业可信度资产）— 不直接收钱，但乘数效应

- **形态**：C（行业可信度）
- **客户**：**不是直接客户**，是 **媒体 / 学术 / 监管机构 / 同行**
- **价值主张**：
  - "**关于'AI 陪伴是否可信'这件事，VZ 是出题人**"
  - 类比：HumanEval 之于 OpenAI Codex，Chatbot Arena 之于 LMSys，HELM 之于 Stanford CRFM——**出题人享受被引用的二阶溢价**
- **变现路径（间接）**：
  1. **抬高 P1/P2/P4 的客单价**——客户做尽调时看到 VZ 在自己定义的 benchmark 上分数最高，是品牌溢价
  2. **吸引头部 LLM 厂商提交跑分**——他们提交反过来成为 VZ 的 PR 资产
  3. **吸引学术合作**（联合论文）→ 招人优势 → 工程能力提升
  4. **未来可能的咨询服务**（"VZ 帮你的产品上 companion-bench 排名"）
- **所需投入**：
  1. 公开域名 + 提交流程 + 排行榜网页
  2. 主动跑一遍 GPT-5 / Claude Opus 4.7 / DeepSeek / Qwen / Llama / Gemini，公开第一份榜单
  3. 1-2 篇技术报告（A1-A6 6 轴方法论 + held-out 协议）
  4. 联合 1-2 个学术机构发学术论文
- **12 个月里程碑**：
  - 公开榜单上线，至少 5 个外部团队提交跑分
  - 至少 1 篇 arXiv preprint
  - 在主流 AI 媒体（机器之心、Synced、AI Coffee Break）至少 1 次报道
- **概率拆解**：60-75%（**这是最高确定性路径**——已有完整 v1.0 实现 + RFC + 私有 held-out submodule；只缺"营销 + 公开化"，所以概率高）
- **Kill criteria**：
  - 12 个月零外部提交 → 设计有问题，需要重做提交体验
  - 头部 LLM 厂商主动质疑方法论且我们答辩不力 → 方法论需要修订（不算砍，是迭代）

### 4.6 路径 P6：Coding Vertical（IDE 集成）— 概率 < 10%，不推荐主推

- **形态**：B 或 A
- **简评**：Cursor 是赛道事实标准，VZ coding vertical 的 `guided_exploration` regime + 负向 recharge 是有趣 idea 但**不构成换工具的理由**。这条路径除非作为 P4 灯塔客户的子能力（"你看 VZ 还能给开发者用"），否则**不应该作为独立商业路径主推**
- **建议**：保留代码、保留学术 / demo 价值（ETA 论文里 coding vertical 的 negative recharge 是 R7 双轨学习的一个有意思证据），但不投入产品化资源
- **Kill criteria**：默认 dormant，只有发生外部强需求（比如有客户主动要求集成 IDE）才激活

### 4.7 路径概率汇总

| 路径 | 形态 | 12 个月签约/上线/收入概率 | 12 个月可达 ARR/收入预估 | 推荐度 |
|---|---|---|---|---|
| P1 Figure-as-a-Service | A+B | 35-50% | 100-300 万 | ⭐⭐⭐⭐ |
| P2 Growth-Advisor | B | 30-45% | 200-500 万 | ⭐⭐⭐⭐ |
| P3 C 端陪伴 App | A | 15-25% | 200-500 万（高烧钱） | ⭐⭐ |
| P4 DLaaS B2B 通用 | B | 25-40% | 100-500 万 | ⭐⭐⭐ |
| P5 Companion Bench | C | 60-75% | 0（间接 ×1.5-2 系数到 P1/P2/P4） | ⭐⭐⭐⭐⭐ |
| P6 Coding | B/A | <10% | <50 万 | ⭐ |

**关键洞察**：P5 是概率最高且**应该最优先做**的路径——不是因为它直接赚钱（不直接赚），而是因为它**作为乘数加在 P1/P2/P4 上**，并且**12 个月内最便宜兑现**。



