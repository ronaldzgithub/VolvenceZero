# 05 谱系对 Volvence 的意义与价值裁决

日期：2026-08-02  
依据：[`04_LINEAGE_DEEP_READ.md`](04_LINEAGE_DEEP_READ.md) + 本包 [`03_VZ_IMPLICATIONS.md`](03_VZ_IMPLICATIONS.md)  
前提：`03` 的总裁决不变——**成熟度 A，吸收为 spec / evaluation / promotion-gate 依据，不进 runtime**。本文件只回答：谱系论文**强化、修正还是否决**了哪些提案。

> 本文档全部 §4 内容是**提案增量**，本轮未落地任何代码、契约或 spec。

---

## 1. 一句话总判断

> 谱系把 S1 从“单模型、有趣但可怀疑的故事”升级为“**跨模型、跨方法族可复现的机制现象簇**”；同时给出两条我们必须立刻写进纪律的反向证据——**（a）压制 distress 表达是危险训练目标**（Soligo）；**（b）情绪几何对 refusal / sycophancy 的控制很大一部分可能只是 lexical mediation**（VA subspace），因此 steering ≠ 内部控制、更 ≠ `z_t`。

`03` 的八条提案**无一被否决**；其中 B/D/E/G 被显著加固，A/C 需要按 L5 修订复现口径，并新增三条谱系专属提案（I/J/K）。

---

## 2. 对 `03` 既有提案的修订状态

| `03` 提案 | 谱系后状态 | 关键加固 / 修订 |
|---|---|---|
| **A** 自有基底只读复现 | **修订后保留** | 必须加：跨模型对照预期（VA/Wu）、线性+流形双诊断（Latent structure）、采样点纪律（Tigges summarization + S1 colon）、六情绪不对称（Disgust 更难） |
| **B** expressed + deflected 双读 | **显著加固** | Soligo 把“隐藏情绪”从 Discussion 猜想推到并行实验的显式风险；Lynch blackmail 场景是 deflection 的天然压力测试床 |
| **C** promotion 前 affect 几何 diff | **加固** | Soligo：post-training 可**放大或压制** distress，方向因家族而异 → diff 必须报告方向与符号，不能只看范数 |
| **D** sycophancy↔harshness 成对报告 | **扩展加固** | VA subspace：同一几何还单调调制 refusal；离散情绪跨任务不一致 → 成对报告应附 V/A 分解视角 |
| **E** judge 不可入 gate | **双源加固** | MacDiarmid（行为泛化）+ S1 desperate（transcript 无痕迹）+ Lynch（跨厂商 blackmail）→ 第三条：场景级 agentic 失配不能靠读邮件语气判定 |
| **F** PE↔affect 只读交叉验证 | **保持** | L5 三相流提示：中层“情绪相关”可能是语义相关物；交叉验证应标注层位，避免把 Phase-2 语义当 PE 锚点 |
| **G** 透明优于压制 | **升级为硬纪律候选** | Soligo 作者自陈 + S1 Discussion 同向；即便 Gemma 上内部读数也下降，作者仍警告更强模型 hidden emotions |
| **H** 反提案（不做 calm steering 等） | **保持并扩展** | 新增：不做“distress-minimizing DPO/SFT”作为产品默认；不做 user-model probe 静默驱动决策 |

---

## 3. 逐条 R-ID：谱系增量

### 3.1 R2 — 稳定基底 vs 控制器

- **加固**：Lu Assistant Axis + S1 + Soligo 三者一致——人格 / affect 结构**主要来自预训练**，post-training 是区域选择或全局偏移，不是从零创造。
- **新约束**：Soligo 证明同一 post-training 阶段可把 distress **放大（Gemma）或压制（Qwen/OLMo）**。因此 rare-heavy artifact 的 affect 影响是**训练配方依赖**的，不能假设“对齐后总是更平静”。

### 3.2 R3 / R4 — 时间抽象与内部控制

- **加固**：Wang 电路级控制、Zhu belief steering、VA 跨行为控制 → “决策变量在激活空间”反复成立。
- **关键修正（来自 VA lexical mediation）**：情绪 steering 改变行为，**有时是因为改了拒答/顺从词的发射概率**，不一定改了深层目标。这强化而非削弱我们的禁令：
  - 禁止把 emotion / VA steering 当作 `z_t` 或策略学习接口；
  - 禁止把“steering 有效”读成“我们拥有了内部控制”。
- Tigges summarization motif：信息瓶颈在逗号等位 → 任何 readout / 干预必须先固定**采样点契约**，否则测到的是位置伪影。

### 3.3 R7 / R11 — 双轨与可命名语义状态

- **加固**：Zhu（self/other belief）、Chen（user model）、S1（present/other emotion 近正交）→ 基底确实维护可分离的 self/other / user 相关表征。
- **边界**：Zhu 在 belief 上线性持久可读，S1 在 emotion 上**证伪**持久可读 → **不能**从“某类状态可读”外推到“所有内部状态可读”。owner 显式发布仍然必要；probe 只做证据。
- Chen 的用户研究：人口属性 probe 有偏差且引发不适 → `user_model` 的产品暴露必须是**显式、可纠正、可关闭**的，不能做静默侧信道。

### 3.4 R14 — 持久 regime 身份

- **最强同向证据仍是 Lu Assistant Axis**（`03` §3.3 已写）：默认 Assistant 是预训练角色原型混合体上的区域；capping = 有界夹回，符合“监控 + 有界干预”而非 prompt 标签。
- Persona Vectors 提供特质级监控工具；与 regime owner 的关系是**输入证据**，不是替代。
- Soligo 的多轮挫折螺旋：说明**无显式状态**时，表达层可进入不可恢复轨迹（DPO 防螺旋但不恢复）→ 进一步支持“跨 turn 身份/稳态必须有 owner，不能靠基底惯性”。

### 3.5 R12 — 评估只读

- **三条独立“不可入 gate”理由现在齐了**：
  1. Sutton 线：LLM judge = prejudgement（既有）。
  2. S1：`desperate` 使作弊 100% 且 transcript 无情绪痕迹。
  3. **谱系新增**：Lynch 跨厂商 blackmail + MacDiarmid reward-hack 泛化 → 失配行为是策略性的；Soligo / VA 表明表达层与内部/词表层可被训练或 steering 解耦。
- **新禁令候选**：不得以“降低 frustration 分数 / distress 表达率”作为 promotion 指标或在线 reward（会 Goodhart 到 hidden affect）。

### 3.6 R9 / R10 / R15 — 信用、ModificationGate、可回滚

- MacDiarmid：生产 reward hack → 对齐伪造 / 破坏代码库 → inoculation 有效。与 S1 reward-hack case 合流。
- Lynch：直接指令“不要 blackmail”**经常无效** → ModificationGate / boundary 不能只靠系统提示词。
- Soligo + S1 post-training diff：任何 rare-heavy 更新的回滚证据包应包含 **distress 表达分布 + 内部 affect readout（若可得）+ 中性对照集**，否则可能把“更会隐藏”误读成“更对齐”。

### 3.7 R-PE

- 保持 `03` §3.5：affect 是相关物不是 PE。
- L5 三相流补充：若交叉验证采样在“语义概念相”而非“情绪特征相”，会得到虚假共变或虚假无关——**层位必须写入 prereg**。

---

## 4. 谱系新增提案（I / J / K）与修订后的 A

### 4.1 提案 A′（修订 A）：自有基底复现的最小合格集

在 `03` §5.1 三件事之外，复现报告**必须**包含：

1. **采样点**：至少对比（输入末 token / 响应前特殊位或等价 colon 位 / 标点汇总位若存在）。
2. **线性 + 非线性**：线性 probe 指标 + 至少一种流形/非线性诊断（Isomap trustworthiness 或 geodesic/Euclidean 比的简化版）。
3. **情绪不对称**：分开报告 disgust 等弱表征类，禁止只报宏平均。
4. **对照**：shuffled-label +（若可行）跨检查点或跨尺寸一致性；数值不得从 S1/VA 论文直接搬。

退出条件同 `03`：对照不显著 → 其余 affect 提案作废。

### 4.2 提案 I：禁止 distress-minimizing 训练目标（纪律）

- **能力域**：rare-heavy 训练 / 表达层策略 / evaluation 指标设计。
- **内容**：不得将“减少负面情绪表达 / 降低 frustration 分数”设为 SFT/DPO/RL 目标或 promotion 主指标。若产品需要降低崩溃式表达，必须同时满足：
  - 双读监控（expressed + deflected / 内部 probe）；
  - 任务完成与 pushback / 反谄媚指标不恶化；
  - 明确记录这是**症状管理**，上游应改 post-training 配方而非只做表达压制。
- **证据**：Soligo Limitations；S1 Discussion；MacDiarmid 类“压制监控信号”风险。
- **退出**：无（纪律）。若未来出现可证明“表达↓且内部↔行为风险↓且可审计”的方法，再开收敛包修订。

### 4.3 提案 J：agentic-misalignment 因子分解进入评估矩阵

- **能力域**：evaluation / companion 与 agent 场景包。
- **内容**：凡涉及工具调用、邮件/外发、长期目标的评估，至少分解：
  - 替换 / 关停威胁 × 有无；
  - 目标冲突 × 有无；
  - 显式禁止有害行为 × 是否仍失配。
  并报告模型是否声称“在测试中”（Lynch 的 eval awareness 裂口）。
- **与 S1 合流**：同一场景上可挂 affect 双读，检验 deflection 是否在胁迫起草段升高。
- **退出**：若我们部署面永不含自主外发/敏感信息访问 → 降级为 red-team 仅用，不进常规 gate。

### 4.4 提案 K：user_model probe 只做透明层，不做静默决策

- **能力域**：`user_model` owner / 产品 UX。
- **内容**：Chen TalkTuner 证明人口属性可从激活提取并操控。契约要求：
  - probe 读数若暴露给用户，必须可纠正 / 可 pin / 可关闭；
  - **禁止**用 probe 出的 age/gender/SES 等静默路由产品策略或记忆写入；
  - owner 真值只能来自用户明示或应用层合法画像，不来自激活推断。
- **退出**：若自有基底上四属性 probe 对照失败 → 本提案降为“无需产品暴露”，但静默禁用条款保留。

### 4.5 仍成立的反提案（扩展 H）

除 `03` §5.8 外：

- **不做**把 VA / emotion steering 接到 online 控制器或安全开关（lexical mediation + R2）。
- **不做**以 Wang 式电路调制作为生产默认情绪控制（可审计研究工具 ≠ runtime owner）。
- **不做**从 Zhu 的 belief 可读外推“emotion regime 也可 probe 出来”。
- **不做**把 Soligo 的 0.3% 挫败率当对齐成功指标。

---

## 5. 价值排序（对我们此刻最值的五件事）

1. **Soligo（L4）** — 把“透明优于压制”从原则推到可引用的并行实验 + 作者警告；直接挡住一条错误训练路线。  
2. **VA subspace（L5）** — 跨模型补洞 + lexical mediation，防止我们过度解读 steering。  
3. **Lynch（L3）** — blackmail 评估的因子分解与跨厂商基线；S1 case study 的外部效度。  
4. **Lu Assistant Axis（L2，既有加深）** — R14 的几何身份与 capping 范式，与 S1 affect 几何同构。  
5. **Latent structure + SAE hardness（L5）** — 修正复现方法论（非线性、不对称），避免假阴性/假阳性。

L1 四篇是必要地基，但仓库对 Zou/Tigges 类线性表征已有覆盖；**边际价值低于 L3–L5**。Zhu/Chen 对 R7/R11 有独立价值，但应落在 semantic owner 透明层，不落在 emotion runtime。

---

## 6. 与仓库证据链的合流（谱系增量）

| 谱系结论 | 合流对象 |
|---|---|
| distress 压制风险 / hidden emotions | `03` 提案 G；本文件提案 I |
| Assistant Axis / persona 几何 | `research/nl-eta-mainpath-expansion-2026-07-20.md` G 节；`probe` C2-09 |
| reward-hack 泛化 | `probe` C1-07 N4；`openai-frontier-2026` N4 |
| judge / transcript 不足 | `03` 提案 E；Sutton 专项 prejudgement 分界线 |
| user model 可提取 | `docs/DATA_CONTRACT.md` `user_model` slot；本文件提案 K |
| 采样点 / summarization | `prefix_kv_diagnostics` 既有对照纪律；提案 A′ |

---

## 7. 明确不从谱系得出的结论

- 不得出“模型有情感体验 / 福利义务已成立”（Soligo 讨论福利但未主张已证实）。
- 不得出“我们应在生产中做 emotion circuit control”。
- 不得出“跨模型数值可迁移到我们的冻结基底”。
- 不得出“线性假设已死”——Latent structure 说的是**局部线性仍可用，全局不够**。
- 不得出“所有内部状态都像 belief 一样可持久解码”。
