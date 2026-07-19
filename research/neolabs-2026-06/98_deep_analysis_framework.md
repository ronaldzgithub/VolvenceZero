# Neo Labs 深度分析框架（PDF-grounded · 三视角）

调研日期：2026-06-13 ｜ 口径见 [`00_roster.md`](00_roster.md) ｜ 下载见 [`_download_summary.md`](_download_summary.md) ｜ 横向综合见 [`99_synthesis_vz_mapping.md`](99_synthesis_vz_mapping.md)

> 本文是 33 家新型实验室的**深度分析方法论 + 跨 lab 反证矩阵 + 局部算法 ROI 台账**。
> 它**补**而非**替** [`99_synthesis_vz_mapping.md`](99_synthesis_vz_mapping.md)：99 是确证导向的横向综合（"四个领域独立印证 VZ 三公理"），本文补齐 99 缺的两条轴——**反证（红队）**与**局部算法借鉴（算法级解耦）**——并把每家落到独立的 `labs/<lab>/analysis.md`（与该 lab 的 PDF 同目录）。
> 对齐 [`../../docs/specs/00_INDEX.md`](../../docs/specs/00_INDEX.md) 的能力域 spec 与 R1–R20 + R-PE 不变量。

---

## 一、为什么需要"三视角"——99 的确证偏误

[`99_synthesis_vz_mapping.md`](99_synthesis_vz_mapping.md) 的核心论证是"收敛"：四个互不通话的社区独立收敛到 VZ 的三条公理（PE 一级信号、冻结基底 + 自适应控制器、latent 空间控制）。这个论证**强但有系统性偏差**：

1. **确证偏误（confirmation bias）**：当你拿着 VZ 的 R 轴去读每篇论文，几乎总能找到"印证"——因为 R 轴本身足够抽象。99 把所有 30 家都读成了"支持 VZ"。但科学性要求：**哪些证据会证伪 VZ？这些 lab 里有没有这样的证据？**
2. **lab 主张层 vs 算法层混淆**：99 在"lab 整体主张"层做映射。但一个 lab 的整体叙事可能与 VZ 对立（如 Ineffable 的"无预训练纯经验"对立 R2），**而其内部的某个局部算法依然可借鉴**。把"叙事"和"机制"绑在一起，会让我们因为反对叙事而错过机制，或因为认同叙事而吞下不该吞的机制。

因此每家 lab 的"与 VZ 的关系"必须拆成**三个独立视角**，且三者结论可以不一致：

| 视角 | 问题 | 判定标准 | 失败模式（缺它会怎样） |
|---|---|---|---|
| **① 确证（先进性背书）** | 这家在哪些 R 不变量上为 VZ 提供了独立证据？ | 必须是**非语言/跨领域**的独立验证才算强背书；同社区复述不算 | 自我循环论证，把"我们的假设"当"被验证的事实" |
| **② 反证（红队）** | 这家在哪里**对立/挑战** VZ 假设？VZ 经得起吗？ | 给出明确裁决：`survives` / `needs-boundary-condition` / `genuine-risk`，并说明边界条件 | 确证偏误；把所有人都读成盟友，丧失证伪能力 |
| **③ 局部算法借鉴（算法级解耦）** | 剥离 lab 叙事后，哪个**具体局部机制**能搬进 VZ？ | 机制 → 目标 VZ spec → 落地动作 → 预期收益 → 风险/前提，五元组缺一不可 | 因反对叙事而错过机制；或因认同叙事而吞下整包 |

**三视角的关键纪律：先反证，后确证。** 默认假设"这家可能证伪 VZ 的某条不变量"，先尽力找反例并裁决；裁决之后剩下的确证才是可信的。局部算法借鉴**独立于**前两者——哪怕一家被判 `genuine-risk`，其某个局部算法仍可能是高 ROI 借鉴。

---

## 二、R 不变量与 VZ spec 对照（分析时的锚点）

每家 analysis.md 的三视角都要落到具体 R 轴与 spec。锚点表：

| R 轴 | 含义 | 对应 spec（`docs/specs/`） |
|---|---|---|
| R-PE | prediction error 是一级原始学习信号 | [`prediction-error-loop.md`](../../docs/specs/prediction-error-loop.md) |
| R1 / R2 / R13 | 多时间尺度；冻结基底 + 自适应控制器；SSL↔RL 交替 | [`multi-timescale-learning.md`](../../docs/specs/multi-timescale-learning.md) |
| R3 / R4 | 时间抽象；控制在 token 空间之上（z_t / β_t） | [`temporal-abstraction.md`](../../docs/specs/temporal-abstraction.md) |
| R5 / R6 | 记忆连续谱；反思与沉淀 | [`continuum-memory.md`](../../docs/specs/continuum-memory.md), [`cms-atlas-titans-uplift.md`](../../docs/specs/cms-atlas-titans-uplift.md) |
| R7 | 双轨（World / Self）学习隔离 | [`dual-track-learning.md`](../../docs/specs/dual-track-learning.md) |
| R8 / R11 / R15 | 快照/契约优先；内部状态可发布；可解释可回滚迁移 | [`contract-runtime.md`](../../docs/specs/contract-runtime.md), [`semantic-state-owners.md`](../../docs/specs/semantic-state-owners.md) |
| R9 / R10 | 层级信用分配；有门控的分层自修改 | [`credit-and-self-modification.md`](../../docs/specs/credit-and-self-modification.md) |
| R12 | 评估覆盖"存在"而非仅任务，且只读 | [`evaluation.md`](../../docs/specs/evaluation.md), [`evidence_program.md`](../../docs/specs/evidence_program.md) |
| R14 | 认知/社交 regime 持久身份 | [`cognitive-regime.md`](../../docs/specs/cognitive-regime.md) |
| R16–R20 | 多人身份 / ToM / 会话角色 / 共同基础 / 群体实体 | [`social_cognition/`](../../docs/specs/social_cognition/) |
| 行动 / 工具 | affordance 选择在 z_t 空间学 | [`affordance.md`](../../docs/specs/affordance.md), [`environment-interface.md`](../../docs/specs/environment-interface.md) |

---

## 三、单 lab `analysis.md` 模板（每家统一遵循）

```markdown
# <Lab 名> — 深度分析

- 分组 / 成熟度 / 一句话主张
- 主要创作者 + 血统（为何与 VZ 共振或对立）

## 1. 核心逻辑（论文级 · PDF-grounded）
逐篇：问题 → 方法/机制 → 关键结果（带 PDF 内具体数字/设置）→ 局限。
（0-PDF lab 用 DOI/链接 + 注明 UNVERIFIED。）

## 2. 与 VZ 的关系（三视角）
### 2.1 确证（先进性背书）
印证了哪些 R 不变量；强调是否为跨模态/跨领域的独立验证。
### 2.2 反证（红队）
挑战/对立了哪些 VZ 假设；逐条给裁决：survives / needs-boundary-condition / genuine-risk；写明边界条件。
### 2.3 局部算法借鉴（算法级解耦）
剥离叙事的具体机制 → 目标 VZ spec → 落地动作 → 预期收益 → 风险/前提。

## 3. 一句话定位
## 附：本地论文清单（同目录 PDF）
```

裁决取值定义：
- **survives**：反例不适用于 VZ 的目标域（关系/EQ/长程养成），VZ 不变量在其边界内成立。
- **needs-boundary-condition**：反例揭示 VZ 不变量成立的**前提**，需在 spec 写明边界（不是推翻，是收窄主张）。
- **genuine-risk**：反例指出 VZ 的**真实风险/盲点**，需要新增设计或评估（进 ROI 台账或风险登记）。

---

## 四、跨 lab 反证矩阵（已回填）

> 每行：VZ 不变量 × 最强反例来源 × 反例论点（PDF-grounded）× VZ 裁决 × 边界条件 / 待办。
> **总判断**：33 家中无一条 R 不变量被**彻底证伪**；但红队暴露了 5 处必须写入 spec 的**边界条件**与 4 处**真实风险**（genuine-risk），并纠正了 99 的 3 处确证偏误（见 §4.1）。这正是本框架相对 99 的核心增量。

| VZ 不变量 | 最强反例来源（lab · 论文） | 反例论点（PDF-grounded） | 裁决 | 边界条件 / 待办 |
|---|---|---|---|---|
| **R2 冻结基底 + 自适应控制器** | Ineffable / Silver（AlphaZero·MuZero） | tabula-rasa self-play、无预训练即超人——"基底可从零长出" | **needs-boundary-condition** | 仅当①完美模拟器②已知规则③廉价无限 self-play④稠密可验证奖励 同时成立时才成立；VZ 关系域四者全缺，故必须保留冻结基底。设为 R2 的"证伪监视器" |
| R2（边界细化） | Profluent ProGen3（IRPO/DPO）· EvolutionaryScale ESM3 · Isomorphic AlphaFold2/3 | 这些"R2 样本"实为端到端/受控全权重微调，纯 DPO 致灾难性遗忘（ppl 8.15→13.87） | **needs-boundary-condition + 纠偏** | 干净 R2 样本应为 ESMFold / BaseFold / Chai-1（冻结基底 + 轻量头）；ESM3/AlphaFold/ProGen3 不算干净 R2，**须修订 99 的 R2★ 标注**；吸收新数据走有界 adapter-delta + KL/正则限幅 |
| R2（叙事祛魅） | Liquid AI（"自适应基底"叙事） | 连续时间网络"自适应"似否定冻结 | **needs-boundary-condition** | 其自适应是冻结权重上的**输入条件化动力学**，非在线权重学习；定位为控制器层 |
| **R9/R10/R15 有界自修改** | Recursive SI（Darwin Gödel Machine）· Sakana（AI Scientist） | 开放式、归档驱动、无外部硬门的自我改进可持续产出——"有界是自缚" | **needs-boundary-condition（照搬开放归档则 genuine-risk）** | 开放式探索只允许在 gate 限定的有界控制器配方空间内；纳入归档必须过**外部非自评**硬门；权重级变更走 rare-heavy + ModificationGate |
| R9/R10（停滞风险） | Recursive SI（AI-GAs "bounded = self-shackling"） | 过度有界会过早收敛、停滞 | **genuine-risk（停滞）** | 在 gate 内引入开放式探索（regret/PE 驱动课程），防过早收敛——有界 ≠ 不探索 |
| **R-PE 不外包** | CZI（rBio 软验证器）· Future House（LLM-judge / ether0） | 用 LLM/世界模型当 reward/judge——"PE 来源可外包" | **needs-boundary-condition（照搬则 genuine-risk）** | 软验证器**仅当**它是 VZ 自身 world/self 预测基底、冻结 + 版本化 + 多源交叉时可接受；外部漂移模型独占奖励 = 不可问责第二 PE 源（rBio GO-掉点即漂移证据；ContraCrow 与人一致仅 60%） |
| R-PE（噪声上瘾） | Skild（Large-Scale Curiosity 的 noisy-TV） | 纯预测误差会对不可减小的随机性上瘾 | **genuine-risk** | PE 必须显式分离 **epistemic（驱动动机）/ aleatoric（不驱动）**，否则系统对用户随机性永远"好奇" |
| **R3/R4 不在 token 空间做长期决策** | humans&（Quiet-STaR）· Recursive SI（COCONUT）· CZI/Future House（token 空间 GRPO） | token 级思考 / 连续 latent 思考 / token 空间 RL 也能学 | **needs-boundary-condition（token 空间 RL 照搬则 genuine-risk）** | 借"可学习内部思考"的**目标**，把**实现**挪到 z_t（Coconut 路线）；token 空间 RLVR 仅限离线 rare-heavy 专科 artifact，禁用于在线关系控制 |
| **R5/R6 显式记忆连续谱（CMS）** | Cartesia（HiPPO/S4/Mamba/Mamba-2） | SSM 的 O(1) 隐状态隐式压缩全历史——"显式 4-stratum CMS 冗余" | **needs-boundary-condition** | SSM 状态不可命名/发布（R11/R8）、不可寻址/无 promotion-decay（R5/R6）、保留策略由 LM 损失而非关系/评估决定（R12）、无 World/Self 与时间尺度隔离（R7/R1）；SSM 只能作 stratum 内压缩器 |
| R5/R6（反向支持） | Arc（Evo StripedHyena + attention hybrid） | 纯有界压缩在精确召回任务上不足，须加注意力 | **needs-boundary-condition（反向支持 VZ）** | 证明"压缩器 + 独立显式召回通道"双件设计的必要性——支持 VZ 保留显式可寻址记忆 |
| **R12 评估先做硬 / 只读 / 非自指** | Periodic（A-Lab 教训）· Sakana（AI Scientist 自评 reviewer）· Lila · Noetik | 自动化闭环可先跑、评估可后补 / 自评即可 | **survives 且强化 VZ** | A-Lab 自主合成结果被学界质疑、AI Scientist reviewer 与 generator 同源——"自动化 ≠ 可验证"；评估必须独立、只读、非自指，先做硬否则放大错误 |
| **R7 双轨隔离** | VERSES（共享生成模型）· 多数生物单轨模型 | 跨 agent 共享生成模型 / 单轨即成功 | **survives / needs-boundary-condition** | 跨 agent 共享 ≠ 单 agent 内 World/Self 合并（经快照发布）；生物单轨模型目标域无 Self 维度，对 R7 沉默非反对 |
| **R11 可命名内部状态** | Cartesia/Liquid/Recursion/Skild（稠密向量）· Stanhope（gauge 等价） | 稠密 latent 不可命名 / 信念坐标不唯一 | **survives / needs-boundary-condition** | 只取压缩器不取不透明向量作运行时语义状态；R11 命名须声明为人为选定的读出坐标系，不宣称等于基底真值 |
| 单一原理 vs 模块化（FEP） | VERSES / Stanhope（自由能单一拉格朗日） | 一条原理统一感知/行动/学习，VZ 的 PE/credit/evaluation 三分是人为 | **needs-boundary-condition** | FEP 是规范性"为什么"、对实现 silent；VZ 三分是 R8 所有权分解 + 防 readout 反噬，属不同层；统一原理恰支持"PE 原始、其余 readout" |

### 4.1 对 99 确证偏误的纠正（红队副产物）

深读 PDF 后发现 [`99_synthesis_vz_mapping.md`](99_synthesis_vz_mapping.md) 有三处把"相关"夸大为"强确证"，建议修订：

1. **R2★ 误标**：99 把 Isomorphic（AlphaFold2/3）标为 R2 强样本（★），但 AlphaFold 是**端到端监督训练、无冻结基底**，消融显示端到端结构梯度是精度关键 → 干净 R2 背书应改引 **ESMFold / BaseFold / Chai-1**（明文冻结基底 + 轻量头）。
2. **R-PE 来源混淆**：99 把生物基础模型的**训练损失（MLM / 对比 / 监督回归 / MMD）**读作 R-PE 背书，属类型错误——它们是**离线训练目标**，非运行时一级 PE。spec 须写明"offline 训练 loss ≠ runtime PE"。
3. **ProGen3 / State 措辞**：99 称 ProGen3"控制器层 DPO"、State"ST 基底 + SE 头"——PDF 显示 ProGen3 是 KL 锚约束下的**全权重受控漂移**（纯 DPO 致灾难性遗忘），State 主力 benchmark **完全不经 SE（无冻结基底）** 且 SE 才是基底、ST 是头（与 99 措辞相反）。

> 含义：99 的"四领域独立收敛"主结论**方向成立**，但**强度被高估**；真正经得起 PDF 级红队的强确证集中在 R2（ESMFold/BaseFold/Chai/Evo/Phenom 冻结基底）、R3/R4（World Models/MuZero/Coconut/π0 latent 控制）、R-PE 理论母体（active inference）与工程算子（ICM）。

---

## 五、局部算法 ROI 台账（已回填）

> 把 33 家 analysis.md 的 §2.3 聚合去重，按**预期收益 × 可落地性**排序。机制五元组（机制 → 目标 spec → 落地动作 → 预期收益 → 风险/前提）。
> 共性前提（适用于绝大多数条目，下表不逐条重复）：①奖励/信号在关系域须用**软验证器**替代标量 reward；②适配限在**控制器层（z_t）**、基底冻结、不做在线端到端梯度；③任何自修改沉淀走 **ModificationGate** 且可回滚；④评估**独立只读**。

### 5.1 高优先（建议本季度反映到 spec / shadow prototype）

| # | 机制 | 来源 lab（论文） | 目标 VZ spec | 落地动作 / 预期收益 | 风险 / 前提 |
|---|---|---|---|---|---|
| 1 | **软验证器 RL**（用冻结预测模型的软概率输出当无硬奖励下的学习信号；软验证 F1 0.66 ≈ 硬验证 0.67） | **CZI rBio** | `prediction-error-loop.md` + `credit-and-self-modification.md` | 用 VZ 自身 World/Self 双轨预测当软验证器，PE=预测与观测响应的分级散度当 reward，在 z_t 空间有界 RL → **给"关系质量无法打分"一个已被生物域验证的原则化解法（全 roster 对 VZ 最独特贡献）** | 验证器须冻结+版本化+多源交叉、见过足量真实关系数据；逐源监控漂移（GO-掉点式），否则成不可问责第二 PE 源 |
| 2 | **epistemic / aleatoric PE 分离 + 内禀好奇（PE=可控特征空间前向预测误差）** | **Skild ICM / Large-Scale Curiosity** + **VERSES EFE** | `prediction-error-loop.md` | PE 显式拆"可减小/不可减小"两路，只用 epistemic 驱动动机 → **从根上防 reward hacking 与对用户随机性上瘾（noisy-TV）** | 需可靠估计两类不确定性；epistemic 估计在 LLM/关系尺度尚属前沿 |
| 3 | **in-context 改进算子（z_t 空间，权重不动）** | **Reflection Algorithm Distillation** | `temporal-abstraction.md` + `multi-timescale-learning.md` | metacontroller 把"会话内如何改进"蒸馏为 z_t 空间 in-context 算子（online-fast 不写权重）→ **天然契合 R2+R3+多时间尺度** | 训练数据需含"学习进展"轨迹（可由 VZ 反思日志构造） |
| 4 | **trust-region/clipped 有界更新（单调改进 + 限幅 + 可回退）+ adapter-delta 判据** | **Thinking Machines TRPO/PPO** + **Profluent IRPO/DPO 双旋钮（β KL 锚 + α NLL 保留）** + **Sakana Transformer² SVF** | `credit-and-self-modification.md` | 给 ModificationGate 一个数学化"自修改限幅 + 不退化保证"算子 + "何时 adapter-delta 足够 vs rare-heavy 重训"的分流判据 | 作用在控制器层、奖励用软验证器、不对 token 策略做 |
| 5 | **有界自修改流水：沙箱 + 归档谱系 + 经验验证门 + 不可改评估器** | **Recursive SI（Darwin Gödel Machine）** | `credit-and-self-modification.md` + `evidence_program.md` | 把 R9/R10/R15 落成可审计、可回滚的 ModificationGate 流水线；纳入归档必须过外部非自评硬门 | 关系轨无客观 oracle，gate 不可放宽；评估器须与生成器异源 |
| 6 | **特征级语用偏好 / RSA 注入 user_model（更稠密 PE 上游提案）** | **humans& Pragmatic Feature Preferences / RSA** | `semantic-state-owners.md` + `social_cognition/02_theory_of_mind.md` | 从"用户为何如此偏好"递归社会推理学 user_model / relationship_state → **样本高效、更准的用户建模** | 作为 PE 上游提案，不替代一级 PE；owner 单写 |

### 5.2 中优先（记入技术路线，shadow 评估）

| # | 机制 | 来源 lab（论文） | 目标 VZ spec | 落地动作 / 预期收益 | 风险 / 前提 |
|---|---|---|---|---|---|
| 7 | **EFE 认知/实用分解 + precision（逆方差）加权门控** | VERSES（active inference） | `prediction-error-loop.md` + `temporal-abstraction.md`(β_t) | 统一"对用户的好奇"与关系目标、原则化平衡探索/满足；提供学习到的非关键词切换门控 | precision 估计需稳定 |
| 8 | **latent 世界模型 + 在"梦"里训练/前瞻控制器** | Sakana World Models · Arc State · CZI TranscriptFormer | `dual-track-learning.md` + `temporal-abstraction.md` | World 轨建冻结生成式 latent 世界模型，控制器在想象 rollout 里低成本训练/短程前瞻 | 监控"梦境作弊"（控制器钻验证器漏洞），保留真实信号校准 |
| 9 | **SSM 有界状态作压缩 stratum + 学习式内容相关遗忘/边界重置** | Cartesia HiPPO/S4/Mamba | `continuum-memory.md` + `cms-atlas-titans-uplift.md` | O(N) 无序列长度先验压缩长程瞬态流；学习式遗忘门控作快照 readout 喂记忆 owner | 须包成 owner 管辖的底层 compressor，决策不外溢为 token 策略 |
| 10 | **检索重排+摘要 + 引文/关系图层级派生索引** | Future House PaperQA2 · Recursion MolPhenix（双冻结基底 + 薄对齐头） | `continuum-memory.md` | 记忆召回更准、token 省 5–10×、降幻觉；跨模态派生索引（数据省一量级） | 派生索引是 owner 管辖的底层组件 |
| 11 | **过量生成→廉价 in-silico 过滤→昂贵真值验证 漏斗 + 证明过滤器确有富集的对照实验** | Latent Labs · Xaira（ProteinMPNN/RFdiffusion） | `prediction-error-loop.md` + `evaluation.md` + `affordance.md` | 从廉价代理到昂贵真值的样本高效漏斗（30–100 而非百万）；把"软验证器是否真富集真值"变成被监控的指标 | 须声明过滤器校准 regime、跨 regime 正交校验 |
| 12 | **CMP 极简消息 schema（confidence + sender-id + should_process bool，禁传内部模型）** | Numenta（Thousand Brains） | `contract-runtime.md` | 用被验证的极简契约强化 R8，并获近免费的跨模块门控（信息显著变化才下传，省算力） | — |
| 13 | **EmbedGEM 双轴 + null 评估法 / 前瞻不可泄漏盲测 + 长度 OOD + 三角色分离** | Insitro EmbedGEM · Inceptive Ribonanza | `evaluation.md` + `evidence_program.md` | 防"表示漂亮但零增量"自欺、防榜单过拟合；跨用户/会话留出 + 混淆校正作 OOD 硬门 | 评估保持只读（R12） |
| 14 | **flow-matching 动作-段 latent 控制 + MAML 控制器层 few-shot + SAC 最大熵防坍缩** | Physical Intelligence π0 / MAML / SAC | `temporal-abstraction.md` + `credit-and-self-modification.md` | latent 段生成（β_t 段闭合）+ 新用户/场景有界少样本适配 + 熵正则防 regime 坍缩 | 段控制是 readout，不外溢为长程 token 策略 |
| 15 | **classifier guidance / 可组合即插即用约束引导（约束=可命名独立 owner）** | Generate Chroma | `affordance.md` + `temporal-abstraction.md` + `contract-runtime.md` | 不动基底权重即做受约束、可增删组合的可控表达；"按需定制行为"降维成"选哪些约束 owner 入场" | 学习式判据"自信≠真实"，引导判据与验收评估须异族（防取悦判据） |
| 16 | **模型选择 = 证据门控的结构变更（慢/罕尺度）** | VERSES / Stanhope（贝叶斯模型比较） | `credit-and-self-modification.md` + `multi-timescale-learning.md` | 把 ModificationGate / R15 回滚变成带退出条件的证据阈值 | 结构改动锁 rare-heavy / 离线 |

### 5.3 低优先 / 设计期纪律 / 概念性

| # | 机制 | 来源 lab | 目标 VZ spec | 说明 |
|---|---|---|---|---|
| 17 | 范畴/类型化快照契约（构造期可检测漂移） | Symbolica | `contract-runtime.md` + `semantic-state-owners.md` | 设计期纪律；禁止宣称运行时 "provable"（position paper、零实证） |
| 18 | Switch top-1 路由 + 可微负载均衡（β_t 专家切换 + 防坍缩） | Periodic（Switch Transformers） | `temporal-abstraction.md` + `affordance.md` | 内容相关、可微、无关键词硬编码；属 rare-heavy 离线扩参方向 |
| 19 | "冻结基底 + 富化条件输入"作一级提升杠杆 + 富化质量门控 | Basecamp BaseFold · Chai-1 · ESMFold | `multi-timescale-learning.md` + `continuum-memory.md` | R2 跨模态合法性论据；输入富化须质量门控（BaseFold 39% 富化无效的教训） |
| 20 | recycling 迭代精修 + 校准置信头（可命名不确定性） | Isomorphic AlphaFold2/3 | `temporal-abstraction.md` + `semantic-state-owners.md` | 控制器提交 z_t 前跑有界 recycling；模块快照自带"预期误差"只读字段 |
| 21 | 闭环 active inference 的 PE 一级性（生物旁证） | Cortical Labs（DishBrain） | `prediction-error-loop.md` | **概念性 · UNVERIFIED（wetware 不可工程化）**；仅作 R-PE 动机旁证 |

---

## 六、进度索引（33 家）

每家深度分析见 `labs/<lab>/analysis.md`（与该 lab PDF 同目录）。分组同 [`00_roster.md`](00_roster.md)。

| 分组 | Lab | 深度分析 |
|---|---|---|
| A 脑启发/神经科学 | Numenta | [labs/numenta/analysis.md](labs/numenta/analysis.md) |
| A | Liquid AI | [labs/liquid-ai/analysis.md](labs/liquid-ai/analysis.md) |
| A | VERSES AI | [labs/verses-ai/analysis.md](labs/verses-ai/analysis.md) |
| A | Stanhope AI | [labs/stanhope-ai/analysis.md](labs/stanhope-ai/analysis.md) |
| A | Cortical Labs | [labs/cortical-labs/analysis.md](labs/cortical-labs/analysis.md) |
| A | Cartesia | [labs/cartesia/analysis.md](labs/cartesia/analysis.md) |
| A | Symbolica | [labs/symbolica/analysis.md](labs/symbolica/analysis.md) |
| B 自主科学家 | Future House | [labs/future-house/analysis.md](labs/future-house/analysis.md) |
| B | Lila Sciences | [labs/lila-sciences/analysis.md](labs/lila-sciences/analysis.md) |
| B | Periodic Labs | [labs/periodic-labs/analysis.md](labs/periodic-labs/analysis.md) |
| B | Recursive Superintelligence | [labs/recursive-superintelligence/analysis.md](labs/recursive-superintelligence/analysis.md) |
| C 生物基础模型 | EvolutionaryScale | [labs/evolutionaryscale/analysis.md](labs/evolutionaryscale/analysis.md) |
| C | Arc Institute | [labs/arc-institute/analysis.md](labs/arc-institute/analysis.md) |
| C | Isomorphic Labs | [labs/isomorphic-labs/analysis.md](labs/isomorphic-labs/analysis.md) |
| C | Chai Discovery | [labs/chai-discovery/analysis.md](labs/chai-discovery/analysis.md) |
| C | Profluent Bio | [labs/profluent-bio/analysis.md](labs/profluent-bio/analysis.md) |
| C | Inceptive | [labs/inceptive/analysis.md](labs/inceptive/analysis.md) |
| C | Latent Labs | [labs/latent-labs/analysis.md](labs/latent-labs/analysis.md) |
| C | Recursion | [labs/recursion/analysis.md](labs/recursion/analysis.md) |
| C | Insitro | [labs/insitro/analysis.md](labs/insitro/analysis.md) |
| C | Generate Biomedicines | [labs/generate-biomedicines/analysis.md](labs/generate-biomedicines/analysis.md) |
| C | Xaira Therapeutics | [labs/xaira-therapeutics/analysis.md](labs/xaira-therapeutics/analysis.md) |
| C | Noetik | [labs/noetik/analysis.md](labs/noetik/analysis.md) |
| C | CZI Virtual Cell | [labs/czi-virtual-cell/analysis.md](labs/czi-virtual-cell/analysis.md) |
| C | Basecamp Research | [labs/basecamp-research/analysis.md](labs/basecamp-research/analysis.md) |
| D 前沿架构 | Sakana AI | [labs/sakana-ai/analysis.md](labs/sakana-ai/analysis.md) |
| D | World Labs | [labs/world-labs/analysis.md](labs/world-labs/analysis.md) |
| D | Physical Intelligence | [labs/physical-intelligence/analysis.md](labs/physical-intelligence/analysis.md) |
| D | Skild AI | [labs/skild-ai/analysis.md](labs/skild-ai/analysis.md) |
| D | Thinking Machines Lab | [labs/thinking-machines-lab/analysis.md](labs/thinking-machines-lab/analysis.md) |
| D | Reflection AI | [labs/reflection-ai/analysis.md](labs/reflection-ai/analysis.md) |
| D | Ineffable Intelligence | [labs/ineffable-intelligence/analysis.md](labs/ineffable-intelligence/analysis.md) |
| D | humans& | [labs/humans-and/analysis.md](labs/humans-and/analysis.md) |
