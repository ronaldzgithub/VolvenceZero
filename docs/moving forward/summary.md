# 《探索方向.md》评估：一份冷静的判断

> 本文件取代了之前那份"90% 信心"的乐观陈述。那份陈述更像团队士气表达，而不是冷静的工程判断。下面这份判断的目标是：**在第 18 个月某个 P0 packet 卡住时，团队不会发现自己的预期建立在没人愿意承认有水分的基线之上。**

## 一、这份文档真正的强项

1. **架构纪律远超同类产品**。R1-R15 + R-PE 不是装饰，它们是逻辑自洽的不变量集合。43 条建议每一条都被强制套进这些不变量，没有让位的余地。这种"先立法、再立法解释、最后写代码"的纪律，目前国内做"数字生命/AI 陪伴"的团队几乎没人能做到。
2. **真的读了论文**。引用的 N1-N8、DM、CMA、A-Mem、AlphaEvolve、CoLA、Persona Vectors 都是 2025-2026 真实存在的工作，不是编造。每条建议至少有一个论文锚点。
3. **方向选择和当前前沿基本对齐**：冻结基底 + 轻量控制器、潜空间 RL、PE-as-signal、Mind/Face 隔离、inoculation prompting、scalable oversight 弱模型 + 工具——这些恰好是 Anthropic / DeepMind / Thinking Machines 这一年内核心安全研究员们正在收敛的设计方向。我们站在了正确的 ridge 上。
4. **SHADOW → ACTIVE + 回滚窗口 + counterfactual evidence** 是产业里真正可执行的工程纪律，不是 demo 文化。

## 二、对原 "90% 信心" 陈述的关键保留

### 1. "Cognitive AGI" 在文档里没有定义

整份文档反复出现"cognitive AGI / digital life"，但没有一处定义"达成"长什么样。是：

- **(a)** 一个在长程关系上明显比 GPT-5 / Claude Opus 4.7 更可信、更稳定、更连续的数字伴侣？
- **(b)** 一个具有"主体性 + 关系连续性"的可商业化数字生命？
- **(c)** AGI 的强义——跨域通用智能匹敌人类？

**(a) 大致 40-55%，(b) 大致 25-40%，(c) 接近 0%**。这份架构不能给 (c)，因为系统能力天花板被冻结的基底模型锁死。控制器层加的是**组织性、持久性、安全性、行为一致性**，不是原始智能。把这份文档当成 (c) 的路径会非常危险——它会让团队和真正在 scaling frontier 的玩家做错误的对比。

### 2. 文档里 "原理免疫" 和 "工程已实现" 的混淆

很多地方写着"VZ 对 N3/N1 描述的失败模式原理免疫"。**这只在不变量被工程证伪前才成立**。OA-2 / OA-6 自己也承认契约测试"待新增"。在 `tests/contracts/test_no_gradient_through_expression_layer.py` 真正写出来并 CI 强制之前，所谓"原理免疫"还只是 spec 的承诺。把"将要做的护栏"当成"已经存在的优势"在叙述里反复出现，会让团队对真实风险脱敏。

### 3. 几个机制级假设其实非常未经验证

| 机制 | 实际成熟度 |
|---|---|
| **Latent Action RL on frozen base (SYS-5)** | CoLA / FR-Ponder 只在受限推理任务上 work；在开放多回合社会对话上的有效性**没人证明过** |
| **PE-as-primary-signal (R-PE)** | 在 closed-world RL 里很好；在开放对话里 PE 的 ground truth 是什么？谁的预测？预测什么？document 没有给出操作化定义 |
| **Persona Vectors / SAE 作为身份漂移监控 (COG-3 / COG-8)** | 这些方向在跨 fine-tune / 跨 seed 上 reproducibility 很差；建立 R14 身份稳定性 evidence 在这个 readout 上风险不小 |
| **9 类 owner snapshot 总线扛得住生产并发** | 没有看到 latency / 显存 / 调度的实测数据 |
| **GNWT-style 多 owner 竞争 (SYS-6)** | CogniPair 是 demo-scale；P2/L 标级是诚实的，但叙述里它仍被当成 "完成数字生命的最后一块拼图" |

这些不是"会失败"，是"还不知道会不会 work，document 把它们当作 yet-to-do 但语气上当成了 will-work"。

### 4. 工程复杂度被低估了

43 条 packet，每条要走"现状核查 → spec 起草 → SHADOW → 评估 evidence → ACTIVE → 回滚窗口"全流程。第 1 波 + 第 2 波就 26 条 P0/P1。如果一条 packet 平均 2-4 周（M 量级），且 packet 之间存在大量依赖（DM-1 → DM-3 / DM-6 / OA-9；SYS-5 → COG-7；EVO-2 → DM-4 → SYS-2…），**单线程下来这是 2-3 年的工程量**。这中间还要养着 lifeform-* 产品线、real-open-dialogue 闭环、audit agent、eval cascade 基础设施。

人力 / 算力没在文档里讨论。

### 5. 评估自身可信度问题（一个递归陷阱）

文档反复强调 evaluation 必须 verifiable / counterfactual / cross-generation。但 `Agentic EQ / 关系长程评估 (COG-11)`、`life-stage curriculum (COG-9)`、`compression quality readout (COG-5)`——这些**本身就需要"我们已经知道什么是好的关系/好的发育/好的压缩"**才能写评估。在 open-ended 任务上，eval 完备性是哲学难题不是工程难题。AlphaEvolve 之所以 work 是因为它的领域（算法、矩阵乘法、调度）**有外生 ground truth**。我们没有。EVO-6 的代际胜率确实是一个 workaround，但它只能告诉你"新一代比老一代好"，不能告诉你"是否在朝 cognitive AGI 收敛"——可能在朝局部 attractor 收敛。

## 三、综合信心标定

| 目标 | 概率估计 | 关键依赖 |
|---|---|---|
| 跑通整套 R1-R15 不变量 + SHADOW/ACTIVE 工程纪律的 monorepo | 70-85% | 取决于团队规模和优先级纪律 |
| 在长程关系、身份连续性、boundary 维护上**显著**优于纯 LLM 产品 | 40-55% | 取决于 COG-1/2/3/11 + CMA-2 + OA-3 真的兑现 |
| 形成"可信赖的数字生命"这一垂直赛道的事实标准 | 25-40% | 取决于市场是否在我们完成之前已被 OpenAI/Anthropic 的 Memory + Persona 功能填平 |
| 实现强义 cognitive AGI / 通用认知智能 | <5% | 架构不是 capability scaling 的对手；substrate ceiling 锁死 |
| 12-18 个月内能拿出第一批 verifiable evidence（不是 demo，是 cross-generation winrate + matched-control eval 通过）证明这条路在真实用户身上有效 | 30-45% | 取决于第 1 波 P0 是否能在 6 个月内完成 |

## 四、能让信心调高的几件事

如果下面几件事在 6-12 个月内成立，对 (a)(b) 的信心可以明显上调：

1. **SYS-2 + OA-3 + DM-4 三件套在真实 lifeform 上跑出 N4 类 misalignment 的复现 + 75% 以上的缓解率**——这是一个外部可验证的 milestone，不是内部 spec。
2. **COG-1 反事实信用分配在真实关系 repair 场景上给出非平凡归因**——不是日志总结，而是"我们能指出哪个 owner 的哪次 snapshot 改变了关系结果"。
3. **PE faithfulness (OA-9) 测出 metacontroller 对 PE upstream 的响应率 > 80%**——证明 R-PE 不是装饰。
4. **CMA-2 VZ-MemProbe 跑出统计显著优于 vector-RAG baseline**——证明 continuum memory 不是过度工程。
5. **第 1 波 11 条 P0 packet 中至少 7 条在 12 个月内 ACTIVE**——证明 convergence-packet workflow 在 43 条规模上不会塌陷。

只要任意 3 件成立，(a) 可以调到 70%，(b) 调到 50% 左右。

## 五、底线判断

这份文档是目前我们手上**最接近"工程级 cognitive architecture 设计"的中文工作**，质量上甩开同行业 80% 的"AI 陪伴 / 数字人"白皮书。它的危险**不在方向错**，而在：

- **叙述上把 spec 的雄心当成已实现的优势**；
- **把"良好架构"等同于"达成 AGI"**。

用一句话替换原 `summary.md` 的开头：

> 这是一条**有可能**做出"可信赖的关系型数字生命"的、**纪律严苛**的架构路线；它**不是 cognitive AGI 的路线图**，它是一个**配得上承载 cognitive AGI 的容器**——能不能装上真东西，取决于 substrate 是否够强、第 1 波 P0 是否兑现、以及团队能否抗住 2-3 年低反馈期的执行压力。

方向**值得**继续走。但请把信心叙述从"90%"改为：

- **在"可信赖的数字生命"目标 (a)(b) 上：40-55% / 25-40%**；
- **在严格 AGI 目标 (c) 上：<5%**；
- 并用上面五条 **milestone evidence** 而不是叙述去维持那个数字。

这是这份评估存在的唯一目的：让我们在 12-18 个月之后回看时，**不会发现整个团队的执行节奏被一个没有人愿意承认有水分的基线绑架**。
