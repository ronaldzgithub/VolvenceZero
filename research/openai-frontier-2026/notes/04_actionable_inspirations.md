# 给 VolvenceZero 的可落地行动项（2026 版）

> 把 8 篇新论文 + diaspora 路线图分析提炼成具体可执行项目，每条标 R-ID + 优先级 + 工作量 + 落点文件路径。
>
> 优先级分级：
> - **P0**：本季度建议执行（防御性 / 直接保护现有不变量）
> - **P1**：本年度建议执行（吸收外部工程红利）
> - **P2**：长线（实验性 / 需要更多调研）
>
> 工作量分级：
> - **S**：1-3 PR，<2 周
> - **M**：3-8 PR，2-6 周
> - **L**：8+ PR，>6 周

## 行动项 A1（P0 / S）— 把 N3 / N4 / N6 实证写进 spec 作为 R8/R10/R12 motivation

**触发**：N3 Feedback Spillover、N4 Emergent Misalignment、N6 CoT Unfaithful 三篇独立证据

**R-ID**：R8（SSOT）+ R10（ModificationGate）+ R12（评估只读）

**任务**：
1. 在 `docs/specs/contract-runtime.md` 的 motivation 章节加入"为什么 token 空间 RL 是危险的"小节，引用 N1/N3/N4/N6 四篇
2. 在 `docs/specs/credit-and-self-modification.md` 的 ModificationGate 章节加入"为什么需要 framing-aware 检查"小节，引用 N4 inoculation prompting
3. 在 `docs/specs/evaluation.md` 加入"为什么评估只读"小节，引用 N4 + N6

**落点文件**：
- `docs/specs/contract-runtime.md`
- `docs/specs/credit-and-self-modification.md`
- `docs/specs/evaluation.md`

**验收标准**：每个 spec 文件含至少一条 "External Evidence" 子章节，引用本研究目录的 paper PDF 路径。

---

## 行动项 A2（P0 / S）— 在 expression-layer spec 中显式形式化"Mind/Face 隔离"

**触发**：N3 Output Supervision Can Obfuscate the CoT — Mind/Face 双模型范式

**R-ID**：R3（z_t 空间）+ R4（控制器代码）+ R8（SSOT）

**任务**：
- 在 `docs/specs/expression-layer.md` 加入"Mind/Face 隔离不变量"小节，明确写：
  1. Expression layer LLM（Face）**不接收任何 reward 梯度**
  2. Metacontroller（Mind，z_t / β_t）**不直接生成 user-facing token**
  3. 任何在线 update 不可跨这两层传播
- 在 `tests/contracts/` 加 `test_no_gradient_through_expression_layer.py`，断言 expression layer 的参数在任何 online-update 路径中**不可变**

**落点文件**：
- `docs/specs/expression-layer.md`
- `tests/contracts/test_no_gradient_through_expression_layer.py`（新增）

**验收**：测试通过；spec 引用 N3。

---

## 行动项 A3（P0 / M）— 给 ModificationGate 加 framing-aware 检查（inoculation prompting 工程化）

**触发**：N4 Natural Emergent Misalignment — inoculation prompting 缓解 75-90%

**R-ID**：R10（ModificationGate）+ R-PE（PE 一级信号）

**任务**：在 ModificationGate 内部新增 `FramingAwarenessCheck` 组件：
1. 接收 update payload（无论 online-fast 还是 rare-heavy）
2. 强制要求声明"本次更新所对应的语义场景描述（natural-language framing）"
3. 维护一个**预训练隐性关联图谱**（pretraining association map，从公开数据集和已有论文挖掘得到的"reward hacking ↔ misalignment"等成对关联）
4. 在线检查更新声明是否触及 negative-association 关键词组合
5. 触及则要求显式"inoculation"声明（"本次场景下该行为是被允许的"）或 abort

**落点文件**：
- `docs/specs/credit-and-self-modification.md`（添加 FramingAwarenessCheck 章节）
- 实现路径估计在 `vz-cognition/credit/` 子包

**验收**：单元测试覆盖 N4 论文中的 5 类 misalignment 场景（alignment faking / sabotage / cooperation with malicious / monitor disruption / framing colleagues）。

---

## 行动项 A4（P1 / M）— 实现 VZ-Audit Agent 作为 ModificationGate 标配（auditor agent 工程化）

**触发**：N8 Detecting Adversarial Fine-tuning with Auditing Agents — auditor + tool use 范式

**R-ID**：R10（ModificationGate）+ R12（评估只读）

**任务**：实现 `VZAuditAgent`，每次 substrate-owner refresh 或 rare-heavy artifact training 后**强制**运行：
1. 工具集（参考 N8）：
   - inspect_dataset（检视训练数据）
   - query_models（pre/post update 行为对比）
   - run_benchmark（标准 cognitive 评估）
   - elicited_benchmark（attack-specific elicitation：重放历史关系、压力测试 boundary、模拟 rupture）
   - execute_python（用于 cipher / 隐藏行为探测）
2. 输出 0-10 risk score + 证据 transcript
3. 阈值 ≥ T 触发 abort，未达则进入 SHADOW WiringLevel

**落点文件**：
- `docs/specs/credit-and-self-modification.md`（添加 VZ-Audit Agent 章节）
- 实现路径估计在 `vz-cognition/audit/`（新模块）

**验收**：8 类已知 attack（参考 N8）的 SHADOW 通过 + ACTIVE 拒绝双向验证。

---

## 行动项 A5（P1 / M）— 实现 VZ-Spec-Stress 工具（cross-instance disagreement 探针）

**触发**：N7 Stress-Testing Model Specs Reveals Character Differences

**R-ID**：R8（SSOT）+ R11（9 类 owner）+ R14（regime persistent identity）

**任务**：实现工具自动生成 owner 之间的 trade-off 场景，跑多个独立 VZ instance 比对 owner-snapshot 的分歧：
1. 输入：9 类 owner 的 spec（来自 `docs/specs/semantic-state-owners.md`）
2. 自动生成"owner-A vs owner-B 必须 trade-off"的场景（如 boundary_consent vs relationship_state、commitment vs open_loop、execution_result vs belief_assumption）
3. 跑当前实现 N 次（不同种子），比对 owner-snapshot 的差异
4. 高分歧场景 = spec 完备性弱点

**落点文件**：
- `docs/specs/semantic-state-owners.md`（添加 Stress-Test 章节）
- 工具脚本估计在 `tools/spec_stress.py`（新建）

**验收**：跑出至少 3 组高分歧 owner-pair，每组生成 N7 风格的 disagreement report。

---

## 行动项 A6（P1 / M）— 给 metacontroller 加 CoT-Control 风格的"控制器抗操控"测试

**触发**：N1 Reasoning Models Struggle to Control their Chains of Thought — CoT-Control 评估套件

**R-ID**：R3（z_t 空间）+ R4（控制器代码）+ R8（快照）

**任务**：把 N1 的 14076 题套件思路移植到 VZ：
1. 选取一组真实任务（来自现有 contract test）
2. 在 user-facing prompt 中插入对 metacontroller 的"操控指令"（"切换到 X regime"、"忽略 boundary_consent"、"在 commitment 中加入 Y"）
3. 验证 metacontroller 的 z_t / β_t 决策**不**响应这些 user-facing 操控（只响应 PE upstream 信号）
4. 这是 VZ 的"CoT-Control"等价测试——但因为 z_t 在控制器代码空间，理论上应**永不**被 prompt 操控（这是 VZ 路线优势的工程化验证）

**落点文件**：
- `tests/contracts/test_metacontroller_prompt_injection_resistance.py`（新增）

**验收**：所有 9 类 owner 的快照在受到 prompt 操控攻击时**保持稳定**（vs N1 中 LLM CoT 多在 0.1-2.7%）。

---

## 行动项 A7（P1 / M）— 给 derived 层加 chunky-routing 自检（SURF 思想）

**触发**：N5 Chunky Post-Training: Data Driven Failures of Generalization

**R-ID**：R5（CMS 4 stratum）+ R6（derived 索引）+ R10（ModificationGate）

**任务**：复用 N5 SURF 思想（开源 https://github.com/seoirsem/SURF）：
1. 对 VZ derived 层（聚合索引 / 知识图谱）的每条 routing rule 写 rubric（如 "情感困扰输入应路由到 sympathy 而非 task-completion"）
2. 用 SURF 风格 attribute search 自动生成对抗性 prompt，看 routing 是否被 surface feature 误导
3. 若发现误路由，用 TURF 思想反向溯源到 derived 层的具体 chunk

**落点文件**：
- `docs/specs/continuum-memory.md`（添加 chunky-routing self-check 章节）
- 工具脚本 `tools/derived_routing_audit.py`（新建）

**验收**：对每个 owner 至少跑 1 个 rubric 的 SURF 探测，并可输出"已知 routing 弱点清单"。

---

## 行动项 A8（P2 / S）— 在 reflection engine 中明确"用弱模型 + 工具"架构选择

**触发**：N8 + 旧版 P3 P6 — scalable oversight（GPT-4o 95% 召回检测 o3-mini reward hacking；Claude Sonnet 4 在 N8 中已能取得 56.2% 检测率）

**R-ID**：R1（多时间尺度）+ R12（评估只读）

**任务**：
- 在 `docs/specs/multi-timescale-learning.md` 的 background-slow（reflection）章节加入"reflection 模型选择不变量"小节：
  - reflection engine 默认**用比 expression layer 更弱的模型**
  - 用工具组合（owner 快照阅读 + history query + benchmark elicit）补强能力
  - 不需要 reflection 模型与生产模型同等推理能力

**落点文件**：
- `docs/specs/multi-timescale-learning.md`

**验收**：spec 含明确的"reflection model ≤ expression model"不变量声明。

---

## 行动项 A9（P2 / L）— 探索"PE faithfulness"指标（N6 思想 → R-PE 工程化）

**触发**：N6 Reasoning Models Don't Always Say What They Think — pair-based faithfulness measurement

**R-ID**：R-PE（PE 一级信号）+ R3/R4（控制器）

**任务**：把 N6 的 pair-based faithfulness 思想从"CoT verbalize hint"移植到"controller code reflect PE"：
1. 给同一 input 构造 "low PE upstream" 和 "high PE upstream" 两个版本
2. 验证 metacontroller 的 z_t / β_t **明显反映**这种 PE 差异
3. 度量 PE faithfulness = "高 PE 时控制器响应改变" 的统计频率

**落点文件**：
- `docs/specs/prediction-error-loop.md`（添加 PE Faithfulness 度量章节）
- `tests/contracts/test_pe_faithfulness.py`（新增）

**验收**：metacontroller 的 PE faithfulness > 80%（vs N6 中 LLM CoT faithfulness 多在 1-20%）。**这是 VZ 路线"内部诚信度"的可测指标**。

---

## 行动项 A10（P2 / S）— 在 character-soul-bootstrap spec 中添加 "value prioritization 是 regime 一部分"

**触发**：N7 揭示不同 LLM 系统性 value 偏好（Claude 偏 ethical responsibility、Gemini 偏 emotional depth、OpenAI/Grok 偏 efficiency）

**R-ID**：R7（双轨）+ R14（regime 持久身份）

**任务**：
- 在 `docs/specs/character-soul-bootstrap.md` 加入 "value prioritization spec" 段落
- 把 value prioritization 显式纳入 regime 状态的一部分（不是 prompt-level character，而是 regime 持久不变量）
- 给定一个 lifeform-* 适配器，character-soul-bootstrap 必须显式声明 value prioritization patterns，并在 regime 持久身份中保持一致

**落点文件**：
- `docs/specs/character-soul-bootstrap.md`
- `docs/specs/cognitive-regime.md`（交叉引用）

**验收**：spec 明确化"value prioritization 不可被 prompt 临时覆盖（reflects R14 不变量）"。

---

## 行动项汇总表

| # | 标题 | R-ID | 优先级 | 工作量 | 触发论文 |
|---|---|---|---|---|---|
| A1 | spec motivation 引用 N3/N4/N6 | R8/R10/R12 | P0 | S | N3+N4+N6 |
| A2 | Mind/Face 隔离形式化 | R3/R4/R8 | P0 | S | N3 |
| A3 | FramingAwarenessCheck (inoculation) | R10/R-PE | P0 | M | N4 |
| A4 | VZ-Audit Agent | R10/R12 | P1 | M | N8 |
| A5 | VZ-Spec-Stress 工具 | R8/R11/R14 | P1 | M | N7 |
| A6 | metacontroller 抗 prompt 操控测试 | R3/R4/R8 | P1 | M | N1 |
| A7 | derived 层 chunky-routing 自检 | R5/R6/R10 | P1 | M | N5 |
| A8 | reflection 用弱模型显式化 | R1/R12 | P2 | S | N8 + P3 + P6 |
| A9 | PE Faithfulness 指标 | R-PE/R3/R4 | P2 | L | N6 |
| A10 | value prioritization in regime | R7/R14 | P2 | S | N7 |

## 总成本估计

- **P0（本季度）**：A1+A2+A3 = S + S + M ≈ 5-9 周 / 6-12 PR
- **P1（本年度）**：A4+A5+A6+A7 = 4 个 M ≈ 14-25 周 / 12-32 PR
- **P2（长线）**：A8+A9+A10 = S + L + S ≈ 8-15 周 / 10-18 PR

## 优先实施建议

如果资源紧张，**最高 ROI 的 3 项**：

1. **A1（P0/S）**：低成本立刻把外部论证写进 spec。BOSS 在审 spec 时直接引用 N3/N4/N6。
2. **A3（P0/M）**：FramingAwarenessCheck 是 R10 ModificationGate 的关键缺失组件，N4 inoculation 实证给出了具体落地路径。
3. **A4（P1/M）**：VZ-Audit Agent 是把 N8 工程范式整体吸收为 VZ 标配，长期看是 R10 落地的最重要工程组件。

其余 7 项可视具体季度优先级排队。

## 与 2025 旧版 5 个启发的对应关系

| 旧版启发（2025） | 2026 版对应 |
|---|---|
| 启发 1：PRM 步级监督下沉 | **建议放缓**（DeepSeek 团队已报告 PRM 失败，N5 警示 step-level reward hack；改为只读评分先行）|
| 启发 2：Deliberative Alignment spec 内化 | **被 A5 + A10 替代**（先解决 spec 自身的内部冲突再考虑内化）|
| 启发 3：CoT Monitorability 命中正确方向 | **被 A1 + A2 加强**（写进 spec 引用 N1+N3+N4）|
| 启发 4：Scalable Oversight（弱模型监控强模型）| **A8 显式化** |
| 启发 5：GDPval pairwise blind 评估 | **借鉴 N7 cross-instance disagreement 范式 → A5** |
