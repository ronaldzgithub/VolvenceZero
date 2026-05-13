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

---

## 5. 推荐路线：12-24 个月的产品化排序

### 5.1 排序原则（不是按 ARR，是按 evidence cascade）

商业路径的排序不应该按"哪条赚钱最多"，而应该按"哪条**最先产出对其他路径有用的 evidence**"。VZ 的特殊性在于：

1. **P5（Companion Bench）产出的"行业排名"是 P1/P2/P4 的尽调武器**——客户做采购评估时，"我们在自己的 benchmark 上最高分"是直接抬高客单价的话术
2. **P1（Figure）产出的"corpus → bundle → adopt → activate 全链可视化 demo"是 P2/P4 的销售视觉资料**——B 端客户能看到"一个完整的'数字爱因斯坦'是怎么 build 出来的"，比 PPT 强 100 倍
3. **P2（Growth-Advisor）产出的"30 天试点客户的运营月报"是 P4 的灯塔案例**——B2B SaaS 销售第一句话永远是"我们已经在 X 客户跑了 X 个月"

所以**正确的排序不是平均分散投入到 4-5 条路径上**，而是**按依赖图串行/并行启动**。

### 5.2 Phase A — 0-6 个月：奠基期

**目标**：把 P5 + P1 demo 推到公开可用，抬高所有后续路径的可信度基础线。

| 时间 | 工作项 | 路径 | 产出 |
|---|---|---|---|
| 0-2mo | Companion Bench 公开域名 + 提交流程 + 排行榜网页 | P5 | 上线 v1 公开页，能接受外部提交 |
| 0-2mo | 自跑 GPT-5 / Claude Opus 4.7 / DeepSeek / Qwen / Llama / Gemini 第一批 | P5 | 第一份"长程陪伴排行榜" |
| 1-3mo | 公开域人物（**爱因斯坦优先**，corpus 充足、英文+德文都有、公共领域）选定 + corpus 全链跑通 | P1 | 端到端可点击 demo |
| 2-4mo | 1-2 篇技术报告 + arXiv preprint（companion-bench 方法论 + figure 复生方法论） | P5+P1 | 学术可引用 |
| 3-5mo | 把 alpha 服务接付款（先支持单租户 PoC 收费）+ 用户管理 dashboard | P1+P2 准备 | 可对外签约的最小服务面 |
| 4-6mo | 接 1 个真实母婴/早教私域客户做 30 天试点（不收费或象征性收费） | P2 | 试点客户的运营月报 v1 |

**Phase A 退出标准**：
- ✅ Companion Bench 公开榜单上线，至少 5 个外部团队提交跑分
- ✅ 至少 1 个公开人物（爱因斯坦）端到端 demo 可现场操作
- ✅ 至少 1 个母婴/早教私域试点跑满 30 天，月报已交付
- ✅ 1 篇 arXiv preprint 公开

**Phase A 不做什么**：
- ❌ 不做 C 端 App（P3）
- ❌ 不做 IDE 集成（P6）
- ❌ 不做 enterprise 销售团队招聘
- ❌ 不做收购 / 融资 / 全国扩张

**Phase A 投入估算**：6-10 人 × 6 个月 = 4-7 人年（团队当前规模可承载）

### 5.3 Phase B — 6-12 个月：变现期

**目标**：把 Phase A 的 evidence 转成 P1/P2 的真实付费收入，验证客单价 / 周期 / NPS。

| 时间 | 工作项 | 路径 | 产出 |
|---|---|---|---|
| 6-8mo | 第二个公开人物（苏轼 / 居里夫人之一） + 1 个学术机构共建 | P1 | 共建客户的 case study |
| 6-9mo | 私域客户从 1 个扩到 3-5 个（席位制定价 2000-8000 元/月/席位） | P2 | 月度续签率 / NPS / churn |
| 7-10mo | Figure 选 1 个**付编译费**的种子客户（博物馆 / 出版机构 / 大学） | P1 | 第一笔实际付编译费的合同 |
| 8-12mo | 灯塔客户接洽（DLaaS B2B 私有部署 PoC） | P4 | 1 个有名字的 PoC 客户 |
| 9-12mo | OEM / 集成方接洽（让其他 SaaS 嵌入 VZ） | P4 | 1-2 个 OEM 谈判桌 |
| 10-12mo | 第二份 Companion Bench 榜单更新（含 GPT-5.5 / Claude 5 等下一代模型） | P5 | 持续行业可见度 |

**Phase B 退出标准**：
- ✅ P1 + P2 累计签约金额 > 300 万人民币
- ✅ 至少 1 个 P2 客户跑满 6 个月并续签
- ✅ 至少 1 个 P4 灯塔客户 PoC 启动（不要求付费，要求 case study 授权）
- ✅ Companion Bench 第三方提交累计 > 20

**Phase B 投入估算**：8-15 人（增加 1-2 个销售/解决方案 + 1 个内容/PR + 维持工程团队）

### 5.4 Phase C — 12-24 个月：规模期 / 路径筛选期

**关键决策点**：到 Phase B 结束时，团队已经有 6-12 个月的真实客户数据和 P1/P2/P4 各自的转化漏斗。**Phase C 的核心动作不是"全力开火"，而是"砍掉 2 条留 1-2 条"**。

| 路径 | Phase B 结束的判定 | Phase C 行动 |
|---|---|---|
| P1 强（>500 万签约 + >2 个客户 + 续签率 > 50%） | 主推 | 行业垂直深化（博物馆 → 教育 → 出版分行业销售） |
| P1 弱（<200 万签约 或 续签率 < 30%） | 砍 / 转弱 | 保留作为 P4 灯塔的子能力，不再独立卖 |
| P2 强（>5 客户 + ARR >300 万 + churn <10%） | 主推 | 行业模板复用化 → 自助化 SaaS（让客户自己上传 profile 自动 compile） |
| P2 弱（<2 客户 或 churn > 30%） | 砍 | 把 growth-advisor 能力下沉到 P4 |
| P4 强（>2 PoC + 至少 1 个付费） | 主推 | 招 enterprise sales 团队 + 私有部署 production-grade |
| P4 弱（无灯塔 PoC 或 PoC 无付费） | 推迟 | Phase D 重新评估 |
| P5 持续 | **永远主推**（成本低，乘数高） | 第三方提交 > 50；至少一个头部模型用 VZ benchmark 解释自家产品 |

**Phase C 不做什么**：
- ❌ 不做 P3（C 端陪伴）除非外部资本主动愿意烧钱
- ❌ 不做 P6（IDE 集成）
- ❌ 不做"我们也搞个 AI Agent SDK 给开发者"——和工程纪律对冲

**Phase C 投入估算**：取决于 Phase B 结果，分两档：
- 乐观（P1+P2+P4 至少 2 条强）：扩到 25-40 人，开始考虑 A 轮融资
- 保守（仅 P5 + 1 条强）：保持 12-15 人，以盈利为目标

### 5.5 排序的反直觉点

很多创业团队会做的事，但 VZ **不应该做**：

1. **❌ 同时启动所有 6 条路径**——团队 10 人左右，分到每条路径上 1.5 人，每条都做不深
2. **❌ 先做 C 端 App 因为"故事好讲"**——P3 是高烧钱赛道，6 个月烧完没结果就死了
3. **❌ 先开发 enterprise 销售团队**——P4 在 Phase A 之前没有 case study，销售开了也卖不动
4. **❌ Phase A 就开始招 PM / Growth / Marketing 团队**——Phase A 的核心是 evidence 输出，不是 GTM

VZ 团队当前的工程纪律（contract runtime / SHADOW→ACTIVE / 不变量 CI）是稀缺资产，**先用工程能力放大已有差异化（P5+P1+P2）**，再用差异化产生的 evidence 倒推商业团队建设，而不是反过来。

---

## 6. 单位经济初稿（Unit Economics）

> ⚠️ 本章数字是**初稿假设**，不是已验证数据。所有数字都标注假设来源，便于 Phase A/B 拿到真实数据后回填。
> 货币单位：人民币（CNY）；substrate 假设：自部署或 substrate-as-a-service，按 GPU 时长折算。

### 6.1 关键成本结构（每个 ai_id × 月）

VZ 是 always-on 数字生命体（tick / scene / followup），**即使用户不主动说话也会有内部计算**。这是和"按调用计费 LLM API"不同的成本结构。

| 成本项 | 假设 | 月成本估算（每个 ai_id） |
|---|---|---|
| Substrate 推理（用户主动 turn） | 每 ai_id 平均 30 turn/天 × 30 天 × 平均 4K token/turn = 3.6M token/月，按 0.005 元/1K token | **18 元/月** |
| Substrate 推理（thinking loop / followup） | 中频 thinking 每天 5 次，followup 每天平均 1 次，合计 +20% turn 当量 | **+3.6 元/月** |
| Memory 存储（scoped + durable） | 平均 50KB/turn × 900 turn/月 = 45MB/月，对象存储 0.1 元/GB | **0.005 元/月**（忽略不计） |
| 慢反思（session-post slow loop） | scene 闭合时跑反思，平均每天 1 次 scene close，每次反思 10 秒 GPU | **3 元/月** |
| Rare-heavy（adapter-delta 训练） | 默认不每月跑；按 1/季度 × 1 GPU-小时摊销 | **2 元/月** |
| Bundle / artifact 存储 | figure bundle 200MB、character profile 5MB | **<1 元/月** |
| Ops（监控 / 日志 / SSE） | 摊到每个 ai_id | **2 元/月** |
| **合计 cost per ai_id × 月** | | **约 30 元** |

**敏感性分析**：
- 如果用户重度交互（每天 100 turn）：成本 ≈ 80 元/月
- 如果客户自带 substrate（client-supplied API key）：VZ 侧成本只剩 5-8 元/月
- 如果走开源 substrate 自部署 GPU pool（Llama / Qwen 32B-72B）：单位 token 成本可降 70%，但需要前置 GPU CapEx

### 6.2 路径 P1（Figure-as-a-Service）单位经济

| 项 | 假设 | 数值 |
|---|---|---|
| **客单价**（首单：编译 + 3 个月托管 + 100 万 token） | 博物馆/大学采购 | 30-80 万人民币 |
| **续费**（年度托管 + 调用阶梯） | 年单 | 10-30 万/年 |
| **Bundle 编译 COGS** | 一次性，含 corpus 处理 + 工程师时长 + GPU（如做 L2/LoRA） | 5-15 万 |
| **托管 COGS**（每客户每月） | 假设 100 万 ai_id × 调用 × 月 ≈ 1 个固定 ai_id 但高强度调用 | 1500-5000 元/月 |
| **毛利率**（首年） | (30 万 - 10 万 编译 - 6 万 托管 COGS) / 30 万 | **~46%** |
| **毛利率**（续费年） | (15 万 - 6 万) / 15 万 | **~60%** |
| **CAC**（学术/博物馆 b2b 客户） | 直接销售 + 行业会议 + 学术合作 | 5-15 万/单（首单） |
| **回本周期** | 首年 | **首年回本，第二年开始净收益** |

**结论**：P1 单笔金额大、毛利率不算高（首年 ~46%）但**续费极稳**（学术机构换供应商成本极高）。是**典型的 long-tail 高 LTV 客户**。

### 6.3 路径 P2（Growth-Advisor）单位经济

| 项 | 假设 | 数值 |
|---|---|---|
| **客单价**（席位制） | 单席位 5000 元/月，平均客户 10 席位 | 5 万/月 = 60 万/年 |
| **L2 编译选配**（如客户提供 reviewed profile） | 一次性 | +20-50 万 |
| **席位 COGS**（每席位每月） | 30 元 substrate + 20 元 ops + 50 元 L1 报表生成 | **~100 元/月/席位** |
| **客户 COGS**（10 席位） | 100 × 10 = 1000 元 + 5000 元客户成功（共享） | **~6000 元/月** |
| **毛利率** | (5 万 - 0.6 万) / 5 万 | **~88%** |
| **CAC**（私域运营总监 b2b） | 行业渠道 / 内容 / 案例 | 3-8 万/单 |
| **回本周期** | (CAC) / (5 万/月 × 88%) | **2-3 个月** |

**结论**：P2 是 **VZ 商业化最理想的单位经济模型**——高毛利、客单价中等、回本快、续约率高的潜力（如果运营月报真能给客户带来 ROI）。

**但**：P2 的关键风险不在单位经济，而在**销售线索来源**。私域运营总监不是上 G2/Gartner 找供应商的，他们看 KOL 推荐 + 同行案例。这意味着 Phase B 必须**先用低价/不赚钱的方式获得 1-2 个有头有脸的灯塔客户**，否则 GTM 转不动。

### 6.4 路径 P3（C 端陪伴）单位经济

| 项 | 假设 | 数值 |
|---|---|---|
| **ARPU**（年订阅） | 600-1200 元/年 | 平均 80 元/月 |
| **每用户 COGS** | 平均 30 turn/天，按 6.1 表 | 30-50 元/月 |
| **毛利率**（不含 CAC） | (80 - 40) / 80 | **~50%** |
| **CAC** | 投放 / 内容 / 渠道，参考 Replika / 国内陪伴产品 | 50-150 元/付费用户（粗估） |
| **D30 留存**（行业基准） | Character.ai / Replika 数据 | 20-30%（付费用户更高） |
| **payback period**（每付费用户） | 50/40 = 1.25 月（CAC 50 时） | **1-3 个月**（如果 CAC < 100） |
| **LTV**（按 D365 留存 ~15%，平均使用 4 个月） | 80 × 4 × 0.5 (毛利) = 160 元 | 实际需要 |

**结论**：P3 单位经济**纸面上能看**，但**只有 CAC < 100 时成立**，而 2024-2026 年陪伴赛道的 CAC 已经被 Talkie / 星野 / 筑梦岛 推到 200-400 元。VZ 没有差异化的 IP 内容库，从 0 起步打 CAC 战是输的。

### 6.5 路径 P4（DLaaS B2B 通用平台）单位经济

| 项 | 假设 | 数值 |
|---|---|---|
| **客单价**（基础授权） | 50-200 万/年 | 100 万/年（中位假设） |
| **专业服务**（首年）| vertical 编译 + 集成 | +30-80 万 |
| **COGS**（私有部署，VZ 提供 substrate） | 100 ai_id × 50 元 × 12 月 = 6 万（假设客户中等规模） | 6 万/年 |
| **客户成功 COGS** | 1 个 SE × 30% 时间 | 20-30 万/年 |
| **毛利率** | (100 - 30) / 100 | **~70%** |
| **CAC**（enterprise 销售） | 销售周期 6-12 个月，含 PoC | 30-80 万/单 |
| **回本周期** | 首年勉强回本 | **12-18 个月** |

**结论**：P4 单位经济**典型 enterprise SaaS**——高客单价、高毛利、长回本。最大成本**不在 COGS**，而在 **enterprise 销售团队建设**（招 1 个有 enterprise 经验的销售总监，年成本 80-150 万），这是 VZ 团队当前**没有的能力**。

### 6.6 跨路径汇总

| 路径 | 客单价 | 毛利率 | 回本周期 | LTV/CAC 估计 | 结论 |
|---|---|---|---|---|---|
| P1 Figure | 30 万首单 / 15 万续费 | 46-60% | <12 月 | 3-5x | **健康** |
| P2 Growth-Advisor | 60 万/年 | ~88% | 2-3 月 | 5-10x | **最健康** |
| P3 C 端陪伴 | 800 元/年 | ~50%（不含 CAC） | 1-3 月（如 CAC 可控） | 1.5-3x（CAC 失控会跌破 1） | **风险高** |
| P4 DLaaS B2B | 100 万/年 | ~70% | 12-18 月 | 3-5x（依赖团队能力） | **大单价但慢** |

**单位经济视角的最终建议**：
- 主推 **P2**（最健康单位经济 + 销售周期相对短）
- 配套 **P1**（高客单价、低 churn、品牌效应）
- **P5 永远跑**（不直接产生收入但乘数效应明确）
- **P4 等 Phase B 结果再决定**
- **P3 / P6 暂不投入**

---

## 7. 进入策略（GTM）

### 7.1 不同路径的销售节奏完全不同

VZ 团队需要在 GTM 上避免犯一个常见错误：**把所有路径用同一套销售方法去打**。下面分路径列出 GTM 关键差异。

| 路径 | 决策链 | 销售周期 | 价值锚点 | 主要 GTM 武器 |
|---|---|---|---|---|
| P1 Figure | 学术机构采购委员会 / 馆长 / 出版人 | 3-9 个月 | "**法务能签字**" | 学术合作 + arxiv 论文 + 文化版面 PR |
| P2 Growth-Advisor | 私域运营总监 / 品牌方 CMO | 1-3 个月 | "**月报让我老板满意**" | 同行案例 + 行业 KOL + 试点免费/低价 |
| P3 C 端陪伴 | 个人决策 | <1 周 | "**它真的懂我**" | 投放 + 内容种草 + 留存设计 |
| P4 DLaaS B2B | 企业 IT/合规/业务多角色 | 6-18 个月 | "**审计能过关**" | 灯塔客户 + 合规白皮书 + 私有部署演示 |
| P5 Companion Bench | 学术 / 工程 / 媒体 | 持续 | "**这个赛道的尺子**" | 公开 RFC + 主动跑头部模型 + 媒体曝光 |

### 7.2 P5 公开化（最先做的事）

**P5 是所有路径的杠杆，所以第一件事就是把 P5 公开化**。具体动作：

1. **域名 + 提交平台**：
   - 注册 `companion-bench.org`（或 `.ai`）域名
   - 排行榜静态页（不需要后端，静态生成即可）
   - 提交流程：fork repo → 跑 `companion-bench run --submission ...` → PR 提交 verdict.json + transcript.md
   - **明确公布私有 held-out submodule 的访问规则**（"加入提交方"、"提交后 30 天解封"等防作弊机制）
2. **第一次主动跑分**（**这是最关键的一击**）：
   - 自跑 GPT-5、Claude Opus 4.7、Gemini 2.5、DeepSeek V4、Qwen3-Max、Llama 4-405B 等 6-10 个头部模型
   - **VZ 自己作为 SUT 也跑一次**——但**不要让 VZ 排第一**（如果排第一会被怀疑作弊，可信度反而崩）；让 VZ 在某些子轴（A3 关系连续性 / A4 自适应学习）领先即可
   - 公开发布"第一份长程陪伴排行榜"，技术报告同步上 arXiv
3. **媒体策略**：
   - 投稿机器之心 / Synced / TechCrunch / The Information
   - 主题不要叫"VZ 出 benchmark 了"，要叫"**长程陪伴时代，闲聊机器人到底谁更可靠？我们建了一个尺子**"
   - 角度是中立第三方的语气，不是产品营销
4. **学术合作**：
   - 联系 1-2 个做对话评估 / 心理学量化的高校实验室（清华 CoAI 组 / 北大 / 港中文 / Stanford CRFM 路线）
   - 共同署名一篇 arXiv preprint
5. **持续性**：
   - 每季度更新榜单一次（含新模型 / 重新跑分）
   - 第三方提交进入 ledger 公开可查

**P5 GTM 成本**：1 个 PR/内容人 × 6 个月 + 域名/服务器 + 头部模型 API 调用费（10-30 万）= 总投入 < 80 万

### 7.3 P1 GTM（学术 / 博物馆 / 出版）

VZ 在 Figure 路径上的 GTM 必须**从工程项目变成文化项目**：

1. **第一案例选爱因斯坦**（不是商业理由，是 GTM 理由）：
   - 公共领域，corpus 充足且多语言
   - 大众识别度高，做 demo 流量大
   - 学术界对"如何复生爱因斯坦"有普遍兴趣
   - **最关键**：完成度高的爱因斯坦 demo 是后续所有 P1 销售的"门票"
2. **第二案例考虑：** 苏轼（中文市场 / 文化 IP）/ 居里夫人（女性 + 法语 / 多语言）/ 鲁迅（中国近代 / 政治敏感性需注意）
3. **学术 PR**：
   - 准备一篇 "Faithful Digital Revival: A Compilation Pipeline for Real-Person AI" 学术 paper
   - 投 arXiv + 投至少 1 个 NeurIPS / ACL workshop
4. **博物馆 PR**：
   - 联系普林斯顿大学（爱因斯坦档案在那里）做共建
   - 联系国内博物馆（孔子博物馆 / 鲁迅纪念馆 / 居住地纪念馆）
5. **法律先行**：
   - 提前请熟悉肖像权 / 著作权 / 公共领域的律师写一份"figure compilation 法律备忘录"作为客户尽调材料
   - 这块法律工作做扎实是 P1 核心壁垒之一

**P1 GTM 成本**：1 个 BD + 1 个法务（外包）+ 学术合作费用 + 公共关系，6 个月内 50-100 万

### 7.4 P2 GTM（私域运营）

P2 不能用 enterprise 销售，要用**行业 KOL 渗透 + 试点驱动**：

1. **找 1 位行业 KOL 做联合发布**（最关键）：
   - 母婴行业找类似"年糕妈妈" / "凯叔讲故事"这种 KOL，或者母婴运营圈里的"运营总监级 KOL"
   - 联合举办 1 场行业沙龙："**用 AI 顾问做私域 LTV，但不要让 AI 变销售**"
2. **30 天试点免费**：
   - 第一批 3-5 个客户给免费 30 天（含数据/月报）
   - 30 天后自动转付费，转化目标 50%+
3. **行业模板包**：
   - 把 `cheng-laoshi` profile 抽出可复制结构 → 让客户自己上传 reviewed profile → 自动 compile
   - 卖"行业模板包"（母婴 / 早教 / 留学 / 财富 4 个起步），每包 10-30 万
4. **月报营销化**：
   - 把 weekly-report 升级成"**给品牌总监看的月度运营报告**"格式
   - 关键指标包装："本月 AI 顾问触发 boundary 拒绝销售 X 次（合规增长）" / "rupture-repair 完成率 X%（用户关系健康度）"
5. **小红书 / 抖音 / 公众号**做 B 端内容：
   - 受众是私域运营总监 / CMO，不是 C 端用户

**P2 GTM 成本**：1 个 BD/CSM + 内容运营，6 个月内 30-60 万

### 7.5 P4 GTM（B2B 私有部署）— 暂缓但需要前置准备

P4 在 Phase A/B 不是主推路径，但需要**提前做 3 件事**避免 Phase C 启动时来不及：

1. **私有部署技术 readiness**：
   - K8s helm chart / Docker compose 离线部署版
   - 离线 license server + 用量统计
   - 灾备 / 监控 / 日志聚合
   - SLO 定义（uptime / latency / data residency）
2. **合规白皮书草案**：
   - 数据流图 / 用户隐私 / 用户可删除路径
   - 等保 / GDPR 兼容性自评（不需要正式认证，但要有书面材料）
   - 国密 / 私有化模型对接指南（兼容 Qwen / 通义 / 智谱 / 文心 / DeepSeek）
3. **PoC 套餐设计**：
   - 30/60/90 天 PoC 模板
   - 客户提供什么 / VZ 提供什么 / SLO 是什么
   - PoC 转付费的转化标准（"如果完成 X 个用户场景且 NPS > Y，则进入付费"）

这 3 件事不是销售工作，是工程 / 法务 / SE 工作，可以在 Phase A 用 1-2 人 × 30% 时间慢慢推进。

### 7.6 渠道 / 合作伙伴策略

VZ 不应该自建所有渠道，应该**找正确的合作伙伴当通道**：

| 路径 | 合作伙伴类型 | 例子（不限定，仅启发） |
|---|---|---|
| P1 | 出版社 / 数字内容公司 / 文旅 IP 公司 | 中信出版 / 三联 / 故宫文创 / 当代敦煌 |
| P1 | 数字人 / 视频公司（提供视觉外壳） | 商汤如影 / 科大讯飞 / 腾讯云数字人 |
| P2 | SCRM / 私域 SaaS 服务商（嵌入式合作） | 微盟 / 有赞 / 企微管家类工具 |
| P2 | 行业 MCN / KOL（联合品牌） | 各垂直行业头部内容机构 |
| P4 | 大型集成商 / 咨询公司 | 普华永道 / 德勤 / 神州数码 / 联想智能 |
| P4 | 国内大厂云市场 | 阿里云 / 腾讯云 / 火山引擎 模型市场 |
| P5 | 学术机构 / 媒体 | 清华 / 北大 / Stanford / 机器之心 / Synced |

合作伙伴策略原则：**给伙伴 30-50% 分成 + 自己留下产品掌控权**。早期不要做"白牌"（白牌等于送掉品牌资产），但可以做"联合品牌"。

### 7.7 国际化与中文市场的取舍

VZ 的工程纪律和 contract runtime 哲学**在中文与英文市场都成立**，但 GTM 路径不同：

| 维度 | 中文市场 | 英文市场 |
|---|---|---|
| P1 Figure | 苏轼 / 鲁迅等本土文化 IP，文旅/出版/教育采购，预算大但慢 | 爱因斯坦 / 居里夫人等公共领域，学术/博物馆采购，国际媒体效应 |
| P2 Growth-Advisor | 私域运营是中文市场强势场景 | 海外私域不发达，要换成"creator economy advisor" |
| P3 C 端陪伴 | Talkie / 星野 / 筑梦岛已卷麻；红海 | Replika / Character.ai 已占住；红海 |
| P4 DLaaS | 国央企 + 大型民企，需求强但合规复杂；周期 12+ 月 | 海外 SaaS 销售更成熟，但 VZ 没有 brand awareness |
| P5 Companion Bench | 中文学术/媒体环境对"benchmark"接受度比西方低 | arXiv + Twitter/X + 海外 AI 媒体是天然战场 |

**建议**：
- **P5 优先英文市场启动**（学术影响力 + 国际媒体）
- **P1 双语启动**（爱因斯坦做英文 demo，苏轼或下一个做中文 demo）
- **P2 主攻中文市场**（私域运营是中文市场强势）
- **P4 中文市场为主**，2 年内不打海外
- **P3 不打**

这是一个**研发驱动公司用工程优势放大渠道差异**的标准打法：研发产出在两个市场都有用，渠道选最适合自己资源的市场。

---

## 8. 关键风险与 Kill Criteria

### 8.1 风险分类

VZ 的风险可分为五类，分别需要不同的应对：

#### 8.1.1 技术风险（Tech Risk）

| 风险 | 可能性 | 影响 | 应对 |
|---|---|---|---|
| Latent action RL 在开放对话上不 work | 中 | 中（影响 SYS-5/COG-7 等论证，但不影响 P1/P2/P5 商业路径） | 不把 latent RL 放进卖点；产品话术只讲已交付能力 |
| PE-as-primary-signal 在开放对话上的可操作化失败 | 中 | 中（影响内核演进速度，不影响商业兑现） | 同上 |
| Persona Vectors / SAE 漂移监控不可复现 | 中 | 低（影响 COG-3/8，不影响产品） | 替代方案：用 R12 family report 的"自我一致性轴"做 readout |
| 慢反思 / 多 vertical 共载在生产并发下 latency 爆炸 | 中-高 | **高**（影响 P2/P4 SLO） | Phase A 必须做生产级 latency / 显存 / 调度实测 |
| substrate 升级（Llama 5 / Qwen4）对 figure bundle 兼容性破坏 | 高 | **高** | 把 substrate compatibility fingerprint 写进 bundle；至少支持 N 和 N-1 两代 substrate |

#### 8.1.2 市场风险（Market Risk）

| 风险 | 可能性 | 影响 | 应对 |
|---|---|---|---|
| OpenAI Memory 升级到与 VZ scoped memory 等价 | **中-高** | 中（影响 P3，不影响 P1/P2/P4 的"治理面"差异化） | 把卖点收紧到"治理 + 多 vertical + 拒答"，不卖"记忆" |
| Anthropic 推出 audit-grade chat | 中 | **高**（直接挤压 P4） | 加快 P4 灯塔客户落地；同时强化 P1+P2（这两个 Anthropic 不会做） |
| 中国大厂（字节/腾讯/阿里/百度）推出"数字人 + 治理"打包 | 高 | 中-高（中文 P4 / P2 受影响） | 走"专业、独立、第三方"定位；不要去和大厂的渠道资源拼销售 |
| 陪伴 / 数字复生的政策风险（监管对生成式 AI 加严） | 中 | **高** | 提前与监管对话；安全话题完整记录；handoff queue + UNSAFE typed enum 是合规材料而不是事后补救 |
| 客户更愿意自建（"我们公司有 AI 团队，自己搭一个"） | 高 | 高 | 把 VZ 定位为"高合规 + 已验证"，不和客户的"自建"竞争效率，竞争**风险预期**和**TTM** |

#### 8.1.3 竞争风险（Competitive Risk）

| 风险 | 可能性 | 影响 | 应对 |
|---|---|---|---|
| 头部数字复生公司（DeepBrain/Storyfile/Hi）吸收 VZ 的 L3+L4 | 中 | 高（挤压 P1） | 把 figure bundle artifact 做成可被引用的标准；占住学术/文化/合规叙事 |
| 国内同行（如类似项目）打 P2 私域 | 中 | 中（挤压 P2） | 保持工程纪律 + 反销售边界 + 月报报表化是壁垒 |
| `companion-bench` 被竞争者 fork 重做并占住生态位 | 中 | **高**（如果 P5 失守，P1/P4 杠杆消失） | Apache 2.0 已让出代码，但 held-out submodule + 持续维护 + 学术合作锁住生态 |
| 大量"数字生命"赛道公司（含套壳 ChatGPT）稀释品牌认知 | **高** | 中 | "工程纪律 + audit + 不变量 CI"是反套壳定位，要主动讲清楚 |

#### 8.1.4 法律 / 合规 / IP 风险（Legal Risk）

这块是 VZ 的**最大隐性风险**，专门展开。

| 风险 | 可能性 | 影响 | 应对 |
|---|---|---|---|
| Figure 复生：在世人物未授权 / 已逝人物家属反对 | 中 | **极高**（一次诉讼足以摧毁 P1 路径信誉） | 严格只做公共领域 + 显式授权 / 法务备忘录提前过；P1 合同里明确权属链 |
| Figure 复生：corpus 中混入未授权出版物 | 中 | **高** | L0 crawl 严格过 license；L1 cleaning 对 license_notice 强制流到 SourceProvenance；L2 verification ledger 不可豁免 |
| Growth-Advisor：医疗/营养建议被认定为非法行医 | 中 | **极高** | `cheng-laoshi` profile 已显式编码"qualitative direction only, no specific brand recommendation"；boundary policy 是合同条款 |
| C 端陪伴：青少年用户接入 / 自伤 / 自杀风险 | 中 | **极高** | closed-alpha 已写"no minors"；任何 C 端如开 P3 必须严格身份核验 + 多重 trip wire |
| 数据出境（海外客户 / 国内 substrate） | 中 | 高 | 私有部署版必须支持完全 air-gap；海外客户走海外 substrate（Together / OpenAI / Anthropic） |
| 用户被遗忘权（GDPR / 中国个人信息保护法） | 中 | 中 | scoped memory + DELETE 路径已实现；但 evidence_root_dir 中的 session evidence 必须有删除路径，目前 closed-alpha 是 minimal scope |

**法律风险的核心结论**：VZ 不能做"一切都用 AI 跑"的产品，必须做"**有合规边界、可审计、可拒答**"的产品。这意味着商业上要主动**放弃**一部分上限——比如 P3 不卖给青少年、P2 不替代医生 / 律师 / 财务师、P1 不做未授权在世人物。**少做一些事换商业的可持续性**是 VZ 的本质策略之一。

#### 8.1.5 团队 / 执行风险（Execution Risk）

| 风险 | 可能性 | 影响 | 应对 |
|---|---|---|---|
| 团队当前是研发驱动，无 enterprise 销售能力 | **高（已知）** | 高（限制 P4） | Phase B 后期招 1 个有 enterprise SaaS 经验的销售总监 |
| 团队无内容 / PR / 媒体能力 | **高（已知）** | 中（限制 P5） | Phase A 招 1 个有 AI 媒体经验的内容/PR 人员 |
| 工程纪律和商业 KPI 之间产生张力（"为客户赶进度跳过 SHADOW→ACTIVE 验证") | 中 | **极高**（破坏护城河） | 工程纪律是合同 SLO 的一部分；任何客户 escalation 升到 CTO/创始人，不让一线工程师承担取舍 |
| 创始人精力被"两个公司"模式分走（研发 + 商业） | 中 | 高 | 明确商业 lead vs 研发 lead 的双头结构；保持每周一次同步而不是合并日程 |
| Phase A 不出 evidence 且团队失去耐心 | 中 | 高 | Phase A 6 个月里至少有 1 次"对外可见的成功"（公开榜单 / arxiv / 爱因斯坦 demo），保持团队士气 |

### 8.2 全局 Kill Criteria（什么情况下应该砍整个商业化方向）

**单条路径 kill 已在第 4 章每条路径下列出**。本节是**整体商业化方向**的 kill criteria——什么情况下应该退回纯研发模式或转型。

| 触发条件 | 评估周期 | 退路 |
|---|---|---|
| Phase A（6 个月）结束时 P5 公开化没出来 + P1 demo 没跑通 + P2 试点 0 客户 | 6 个月 | **降速**：保留 1-2 人做基础设施维护，其他人退回研发 |
| Phase B（12 个月）结束时累计真实付费收入 < 50 万 | 12 个月 | **重定位**：不再追求商业化，转型纯研发 / 学术 / 开源项目 |
| 12-18 个月内 OpenAI 或 Anthropic 推出与 VZ 三大差异化（治理 / 拒答 / 多 vertical）等价的官方功能 | 持续监测 | **接受现实**：转型纯学术 / 教育 / 文化 IP 这类大厂不愿做的 niche |
| 团队核心 1-2 人离开 | 不可预测 | 立即评估剩余人员能撑住的最小路径，砍其它 |
| 法律风险事件（一起重大投诉 / 媒体负面 / 监管约谈） | 立即 | 暂停所有相关 vertical 销售；做完整复盘后再恢复 |

### 8.3 推荐的全局健康度仪表盘

每月评估一次，不健康超过 2 个月触发整体战略评审：

| 指标 | 健康阈值 | 来源 |
|---|---|---|
| Phase A 进度（按 §5.2 退出标准的完成度） | 每月 +15% | 项目管理 |
| 工程纪律违规（绕过 SHADOW→ACTIVE / 跳过 contract test 上线）次数 | 0 | git log + CI report |
| Companion Bench 第三方提交数 | 月度增长 > 20% | 公开榜单 |
| Figure demo 公开访问 PV | 月度增长 > 10%（Phase A 后期） | analytics |
| 私域试点客户的 NPS 或 CSAT | > 8/10 | 客户调研 |
| 团队人员流失 | 0/月 | HR |
| 现金跑道 | > 12 个月 | 财务 |
| 合规事件（法律 / 内容 / 安全） | 0 严重事件 / < 3 中等事件/季度 | 合规 dashboard |

任何指标连续 2 个月不健康 → 战略评审（不一定是砍，但必须重新看）

---

## 9. 12-18 个月可写 OKR 草案

> ⚠️ 本章是 OKR 草案，**不是合同**。OKR 需要团队整体认领后才生效；本章节只给一个草稿框架供讨论。

### 9.1 公司级 Objective（12-18 个月）

**O：把 VZ 从"工程上有差异化的研究系统"变成"长程关系型代理这一新品类的事实标准持有者"。**

这个 O 不写"做到 X 万 ARR"。原因：

1. ARR 在 12-18 个月内对 VZ 这种深度技术驱动的项目**不是 leading indicator**——你做对所有事情但市场需要时间消化
2. 把"事实标准持有者"作为目标，能引导团队优先做 P5（公开 benchmark）+ 有名字的灯塔客户，而不是追短期收入
3. 这个 O 是**双向可证伪的**：12 个月后看"有没有外部团队/媒体把 VZ 当成长程关系型代理的参考实现 + 有没有第二条客户路径产生续费"

### 9.2 关键结果（KR）

| KR | 量化目标 | 对应路径 |
|---|---|---|
| KR1：Companion Bench 公开榜单上线，至少 5 个外部团队提交 | 6 个月内 | P5 |
| KR2：至少 1 篇 arXiv preprint（companion-bench 或 figure 复生方法论），至少 1 次主流 AI 媒体报道 | 9 个月内 | P5 |
| KR3：至少 1 个公开人物（爱因斯坦）端到端可点击 demo 公开访问 | 6 个月内 | P1 |
| KR4：至少 1 个 P1 签约客户付编译费（含象征性首单 PoC） | 12 个月内 | P1 |
| KR5：至少 3 个 P2 签约付费私域客户（席位制），其中至少 1 个续签 6 个月以上 | 12 个月内 | P2 |
| KR6：至少 1 个有名字的 P4 灯塔 PoC（不要求付费，要求授权 case study） | 18 个月内 | P4 |
| KR7：累计真实付费收入 > 300 万人民币 | 18 个月内 | 综合 |
| KR8：工程纪律不变量：核心 contract test（含 import boundaries + 不变量 R1-R15）零回归 | 持续 | 全部 |
| KR9：合规零重大事件（无法律诉讼 / 无监管约谈 / 无重大用户安全事故） | 持续 | 全部 |
| KR10：核心团队人员流失 0 | 持续 | 全部 |

### 9.3 反 OKR（不该作为 KR 的事）

为了避免战略漂移，明确写出**不应该作为 KR**的指标：

- ❌ "DAU / MAU"（除非 P3 启动）
- ❌ "代码行数 / commit 数 / merge PR 数"
- ❌ "vertical 数量从 5 增到 8"（数量不是价值，深度才是）
- ❌ "GitHub stars / 社区规模"（companion-bench 例外）
- ❌ "融资金额 / 估值"（融资是手段不是目的；只在确实需要时融）

### 9.4 OKR 评审节奏

- **每 2 周**：内部进度同步（不是 OKR 评审，是项目管理）
- **每 90 天**：OKR 评审 + 本商业评估文件更新（见第 11 章）
- **每年**：全公司战略评审 + OKR 重写

---

## 10. 不应该做的事（Anti-goals）

把"不做什么"写下来比"做什么"更重要——这是创业公司**不被机会拖死**的护身符。下面这些事即使有机会也**不应该做**，至少在 24 个月内不应该做。

### 10.1 产品反目标

| 不做 | 原因 |
|---|---|
| ❌ 自研基础大模型（pretrain from scratch） | 与 R2 冻结基底反向；substrate ceiling 是赛道选择 |
| ❌ 做通用 AI Agent 框架 / 开源 SDK 给开发者 | 与 contract runtime 哲学冲突；Dify/LangChain 已占住 |
| ❌ 做 AI 数字员工 outbound 销售（"AI 替代销售员"） | 与 boundary policy / no-hard-sell 反向 |
| ❌ 做未成年人陪伴产品 | 法律 / 伦理 / 公关风险极高 |
| ❌ 做 AI 心理咨询 / 医疗诊断 / 法律意见 / 财务建议 | 牌照 + 责任风险；只能做 triage |
| ❌ 做"赛博伴侣 + 氪金催化"商业模式 | 与价值观冲突；和 boundary policy 设计反向 |
| ❌ 做未授权在世人物的数字复生 | 法律 + 道德双重雷 |
| ❌ 把 VZ 的 internal slot / owner schema 暴露给第三方开发者 | 破坏 R8 SSOT；外部依赖会变成迁移枷锁 |

### 10.2 商业反目标

| 不做 | 原因 |
|---|---|
| ❌ 在 Phase A 招 enterprise sales team | 没有 case study 销售开了也卖不动 |
| ❌ 在 Phase A 起 PR / Marketing 大编制 | 没有产品成熟度，PR 出去会反噬 |
| ❌ 接受改变工程纪律的客户合同（"我们急着上线，跳过 SHADOW 验证") | 破坏护城河 |
| ❌ 接受让 VZ 变成"客户 OEM 白牌"的合同（除非分成极高且短期） | 送掉品牌资产 |
| ❌ 跨多条路径同时投入超过 30% 的研发力量 | 团队规模不允许；散打必败 |
| ❌ 在中国市场做政治敏感人物的数字复生（无论"红"或"黑"） | 政治风险 |
| ❌ 接受不能 air-gap 的中国央国企 / 政府客户 | 数据出境 / 合规雷 |
| ❌ 公开 companion-bench 的私有 held-out 提示集 | 一旦泄露，benchmark 价值归零 |

### 10.3 文化反目标

| 不做 | 原因 |
|---|---|
| ❌ 把"AGI"作为公司叙事 | 见 `summary.md` §1，会让团队和媒体对赌错误的 baseline |
| ❌ 用"我们能让 AI 真的有情感 / 有意识"作为营销话术 | 不可证伪，且伦理上不诚实 |
| ❌ 用"超越 GPT/Claude"作为竞品话术 | substrate ceiling 锁死，会被实测打脸 |
| ❌ 用"完全替代真人陪伴 / 真人顾问"作为话术 | 与 boundary policy + handoff queue 设计反向 |
| ❌ 在 PR / 商务材料里夸大已交付能力（把 spec 当成已实现） | 见 `summary.md` §2 反面教材 |

### 10.4 这一章的核心结论

**VZ 商业化的最大风险不是市场没有需求，而是团队会被太多"看起来不错的机会"分散**。  
工程纪律的对偶（dual）是商业纪律——**学会拒绝**比**学会签约**更重要。  
这一章应该作为每次商业决策的 first-pass filter：**任何提议如果命中本章，默认拒绝；要做必须有书面例外审批**。

---

## 11. 复盘机制：每 90 天回看本文件应该改什么

### 11.1 为什么需要复盘机制

本评估是**草稿 v0.1**，不是金科玉律。原因：

1. 6 个月后市场已经变了（OpenAI / Anthropic 会发布新东西）
2. 6 个月后 VZ 自己会变了（P5 上线后 evidence 会重写差异化叙事）
3. 6 个月后团队的真实执行能力数据出来了（Phase A 退出标准达成情况会校准 Phase B 假设）

### 11.2 90 天复盘清单

每 90 天回看本文件，按下列清单逐项判断**应不应该改**：

| 章节 | 复盘问题 |
|---|---|
| §1 系统的商业本质 | 经过 90 天客户对话，"我们卖的是什么"的描述是否需要更新？哪些用词被客户秒懂？哪些用词客户听不懂？ |
| §2 差异化资产盘点 | Tier-1/2/3 是否需要重新分层？某些 Tier-2 是否真的能 6 个月成熟？某些 Tier-1 是否在客户面前其实不可见？ |
| §3 市场结构 | 大厂 / 同行有没有发布新功能挤压差异化？时间窗口估计是否过乐观？ |
| §4 路径备选 | 每条路径的概率估计回调多少？kill criteria 是否触发？是否有新路径出现？ |
| §5 推荐排序 | Phase A 退出标准达成情况；是否需要把 Phase B 提前/推迟？ |
| §6 单位经济 | 真实客户数据回填后客单价 / COGS / CAC / 回本周期是否与假设一致？ |
| §7 GTM | 实测中哪些渠道有效 / 无效？合作伙伴策略是否需要调整？ |
| §8 风险 | 是否有新的风险出现？哪些风险已经消退？ |
| §9 OKR | KR 完成度；是否需要重写下一季度 KR？ |
| §10 反目标 | 是否有反目标被破坏？为什么？ |

### 11.3 复盘的输出格式

每次复盘后，**不直接 overwrite 本文件**，而是新建：

- `docs/business/commercialization-review-YYYY-MM-DD.md`：本次复盘的判断与变更建议
- 同步更新本文件的 `Last updated` 与版本号（v0.1 → v0.2 → ...）
- 重大方向变化（比如砍掉一条路径）→ 同步更新 `docs/prd.md` §3.2 use case 表 + `archetecture.md` 相关 vertical 边界

### 11.4 复盘者

每次复盘至少包含 3 个角色：

- **创始人 / CEO**（最终判断）
- **CTO / 工程 lead**（工程现实校准）
- **商务 lead / BD**（市场现实校准）

如果团队还没分化出 3 个角色，至少 2 个角色 + 1 个外部顾问。

---

## 附录 A. 名词对照表

| 名词 | 含义 |
|---|---|
| VZ | VolvenceZero（本仓库）的简称 |
| substrate | 冻结的基础 LLM（Llama / Qwen 等开源；可选 OpenAI / Anthropic 等闭源 API） |
| owner | 一个运行时区域的唯一所有者（R8） |
| slot | 模块发布的不可变快照位置（见 `docs/DATA_CONTRACT.md`） |
| vertical | 一个 lifeform-domain-* wheel，编码"这只生命体在乎什么"的 drive 集合 + 知识/案例/策略/边界 |
| DLaaS | Digital Life as a Service，VZ 的多租户控制平面（见 `docs/specs/dlaas-platform.md`） |
| Companion Bench | VZ 自发布的开源长程陪伴评估基准（见 `packages/companion-bench`） |
| L1-L4（figure） | 真实人物数字复生的保真阶梯：L1 语气 / L2 立场 / L3 引证 / L4 拒答 |
| 7 天 playbook | growth-advisor vertical 的私域顾问 7 天养成阶段化策略 |
| ARR | Annual Recurring Revenue，年度重复性收入 |
| CAC | Customer Acquisition Cost，获客成本 |
| LTV | Customer Lifetime Value，客户生命周期价值 |

## 附录 B. 关键引用文件

- `docs/prd.md` — 产品需求文档（§3 目标用户与场景，§5 能力域，§10 里程碑）
- `archetecture.md` — 25 wheel × 3 层边界 charter
- `docs/moving forward/summary.md` — 团队对 (a)/(b)/(c) 三档目标的冷静自评
- `docs/closed-alpha-api-service.md` — closed-alpha 服务面（已上线）
- `docs/specs/dlaas-platform.md` — 多租户控制平面
- `docs/specs/figure-vertical.md` — 真实人物数字复生 vertical
- `docs/specs/companion-bench.md` — 外发开源 benchmark
- `packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profiles/cheng_laoshi.py` — `cheng-laoshi` reviewed profile

---

## 变更日志

- 2026-05-13：v0.1 初稿。基于仓库截至 2026-05-10 的状态做的全面商业评估。下次复盘 2026-08-13（90 天）。








