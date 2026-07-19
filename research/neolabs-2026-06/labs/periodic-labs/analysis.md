# Periodic Labs — 深度分析

- **分组 / 成熟度**：B 自主 AI 科学家 / 闭环发现 ｜ 成熟度低（2025 出 stealth，尚无第一方可验证论文；本目录 PDF 全为创始人奠基工作，非公司本名成果）
- **一句话主张**：把物理实验室当作 RL 环境——AI 设计材料 → 机器人自主合成 → 真实测量反馈 → 闭环，靠专有/负样本数据飞轮发现新材料（首攻高温超导）。
- **主要创作者 + 血统**：William (Liam) Fedus（联创，前 OpenAI VP Research，**Switch Transformers** 一作）、Ekin Dogus Cubuk（联创，前 DeepMind 材料/化学负责人，**AutoAugment** 一作、**GNoME/A-Lab** 关键人物）、Alexandre Passos 等。
- **为何与 VZ 共振 / 对立**：表层叙事（"自然即 RL 环境"、闭环数据飞轮）共振于 R1/R13（SSL↔RL 交替）与 R-PE（真实测量作为预测误差）；**但本 lab 是全 roster 关于"自动化 ≠ 可验证"的最清晰教训来源**——其引用的 A-Lab 自主合成结果在学界被质疑。本分析以**反证为重心**，把这条教训发展为 VZ R12（评估必须先做硬）的正向论据；同时把两篇创始人奠基工作的**局部算法**与"公司叙事"严格解耦。

## 1. 核心逻辑（论文级 · PDF-grounded）

> **重要切分**：本目录 2 篇 PDF 是**创始人**在视觉/NLP 上的奠基工作，与公司的"自主材料实验室"主张**不是同一回事**。公司主张所依赖的 GNoME / A-Lab 论文付费、且 A-Lab 结果有争议（本地无 PDF），故单列为 **1.B 公司主张（UNVERIFIED）**。

### 1.A 创始人奠基工作（PDF-grounded）

#### Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（2101.03961, Fedus/Zoph/Shazeer, 2021/JMLR 2022）
- **问题**：稠密 Transformer 对所有输入复用同一套参数，扩参=扩算力，代价昂贵；MoE 虽能"为每个输入选不同参数"实现常数算力下的超大参数量，但因**复杂度、通信成本、训练不稳定**而难以普及。
- **方法/机制**：**简化稀疏路由为 top-1（"Switch" routing）**——路由器 `Wr` 对 token 表示 x 产生 logits `h(x)=Wr·x`，softmax 得 N 个专家的门控值 `pi(x)`；与"k>1 才有可学路由梯度"的旧观念相反，**每个 token 只路由到单个最高概率专家**，输出 `y = pi(x)·Ei(x)`。`pi(x)` 保留路由器可微性。引入 **expert capacity =(tokens/experts)×capacity_factor**，溢出 token 经残差直接跳过（dropped）。**可微负载均衡辅助损失** `loss = α·N·Σ fi·Pi`（fi=分派给专家 i 的 token 比例，Pi=路由器分给 i 的概率均值），在均匀路由下最小化，α=10⁻²。配套：路由器输入选择性 cast 到 float32 稳定 bfloat16 训练、缩放初始化、专家 dropout。
- **关键结果（PDF 内）**：与算力匹配的 T5 相比预训练 **7×+ 提速**（FLOPs/token 相同）；做到 **1.6T 参数（Switch-C，2048 专家）**，比 T5-XXL **4× 提速**；101 种语言**全部**获益（平均 5×，91% 语言 ≥4×）；稀疏→稠密蒸馏可缩 99% 体积仍保 **~30%** 质量增益；dropped token 通常 **<1%**。
- **局限**：**硬切换路由带来训练不稳定**（Switch-XXL 高 FLOPs 版偶发不稳，需早停；1.6T 的 Switch-C 反而稳定）；**上游困惑度增益未充分转化到下游**（知识类任务受益、推理类滞后）；专家**同质**（未来才考虑按难度路由到异质大专家）；本质是"参数量"这一**纯扩展轴**，路由是为扩展服务，非为"在线行为切换"设计。

#### AutoAugment: Learning Augmentation Strategies from Data（1805.09501, Cubuk/Zoph et al., 2018）
- **问题**：数据增强长期靠**人工设计**、数据集特定、跨集不迁移（如水平翻转利于 CIFAR 但伤 MNIST）；"自动学习增强策略"是公认未解问题。
- **方法/机制**：把"找最优增强策略"建模为**离散搜索**。**控制器 RNN（单层 LSTM，100 隐元）采样一个 policy** = 5 个 sub-policy，每个 sub-policy = 2 个图像操作，每个操作带（类型、应用概率、幅度）；操作来自 PIL 16 种 + Cutout/SamplePairing，搜索空间 ≈2.9×10³²。采样的 policy 用来训练**固定架构的 child 网络**至收敛，其**验证准确率 R 作为奖励**回传；R 不可微，故用**策略梯度（PPO）**更新控制器。每数据集采样约 15,000 个 policy，最后拼接最优 5 个为 25 sub-policy 终训。
- **关键结果（PDF 内）**：SOTA——CIFAR-10 误差 **1.5%**（↓0.6）、CIFAR-100 **10.7%**、SVHN **1.0%**、ImageNet top-1 **83.5%**（↑0.4）；**策略可迁移**：ImageNet 学到的 policy 迁到 FGVC 数据集普遍提升（Stanford Cars ↓1.2、Aircraft ↓1.8），且从不损害性能；学到的策略**因数据集而异**（CIFAR 偏色彩变换；SVHN 偏 Invert/Shear 等几何变换，与"数字相对色/天然倾斜"先验吻合）。
- **局限**：**搜索昂贵**（ImageNet 约 15,000 GPU·hr）；**消融显示 RL 非必需**——随机采样策略也能改进（3.0% vs AutoAugment 2.6%），作者明言"主要贡献是搜索空间与方法，不是离散优化器"；产物是**离线学到的固定 policy 工件**，非在线适应；奖励是一个**清晰可验证的下游验证准确率代理**。

### 1.B 公司主张（UNVERIFIED · 付费/争议）
- **GNoME（Nature 2023, doi:10.1038/s41586-023-06735-9）**：GNN + 主动学习预测 ~220 万稳定晶体（付费，本地无 PDF）。
- **A-Lab（Nature 2023, doi:10.1038/s41586-023-06734-w）**：闭环机器人自主合成，宣称 17 天合成 41 种新化合物。**该结果发表后被多位材料学家公开质疑**（如 Robert Palgrave 等指出 X 射线表征不足、部分产物被误判/未真正合成）。这是公司"自主实验室"叙事的核心依据，但**其可验证性正是争议焦点**——构成本分析 §2.2 的 headline 反例。

## 2. 与 VZ 的关系（三视角）

> **本 lab 重心在 §2.2 反证**：A-Lab 教训是全 roster 最清晰的"自动化 ≠ 可验证结果"案例，直接服务于 R12（评估先做硬）。先反证、后确证。创始人论文的局部机制在 §2.3 与公司叙事彻底解耦。

### 2.1 确证（先进性背书）
- **R1 / R13（中，但仅限叙事层）**：公司"模型设计→机器人合成→真实测量→闭环"是"压缩（世界模型/主动学习）↔强化（实验验证）"交替的工程范例，真实物理测量天然是不可外包的 PE 来源 → [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md)、[`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md)。**注意：此条为 UNVERIFIED 叙事级背书，证据强度弱于 Cartesia/EvolutionaryScale 等跨模态硬验证。**
- **R3/R4（弱-中，机制级）**：Switch 的 top-1 路由是一个**内容相关的离散选择器**（按 token 表示选专家），它独立验证了"用一个学到的门控在一组有界子模块间切换"在工程上稳定可扩展——这是 β_t/affordance 式切换的**机制对照**（详见 §2.3 的精确界定）→ [`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md)。
- **R9/R10（弱）**：用**真实物理现实**作为自修改的硬验证门控，方向上呼应"自修改要有可验证证据"——但这恰被 A-Lab 争议反噬（见 §2.2）→ [`credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md)。

### 2.2 反证（红队）

**反证 A（headline · 全 roster 最清晰教训）：自主科学家闭环证明"先把自动化跑起来、评估可以后补"，因此 VZ 坚持 R12（评估先做硬）+ R-PE（PE 来源不外包）是自缚。**
公司叙事把"自然即 RL 环境 + 机器人闭环 + LLM 当任务/奖励生成器"包装成"自动化越快越好"。A-Lab 正是这条路线的旗舰：宣称 17 天自主合成 41 种新化合物。但其结果被学界质疑——**当合成产物的表征/验证不过硬时，自动化只是更快地放大了未经证实的"成功"**（41 种里有多少真的是新相、真的合成成功，成为争议核心）。
- **裁决：对 VZ 而言 survives（且强化 R12）；对跳过评估的任何闭环而言 genuine-risk。** 反例的"教训方向"恰好反转：它不是"评估可后补"的证据，而是"评估不先做硬 → 自动化放大错误"的实证。VZ 的 R12（评估覆盖存在、只读、自动化前先做硬）在此被**正向印证**。
- **残余 genuine-risk（必须登记）**：若把 **PE/奖励的来源外包**给一个自动化标注器或 LLM 任务生成器（A-Lab 用自动相分析判定"是否合成成功"），一旦该判定器偏了，整个飞轮一起偏且**无法检测**——直接违反 R-PE"内禀 PE 不外包"。
- **边界（写入 spec）**：(1) 任何自修改/学习闭环上线**前**必须有可验证、可对照、只读的 eval（R12 hard-first）→ [`evaluation.md`](../../../docs/specs/evaluation.md)、[`evidence_program.md`](../../../docs/specs/evidence_program.md)；(2) PE 的接地信号不得外包给一个未经独立验证的自动判定器 → [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md)。

**反证 B：Switch 用"参数量独立于 FLOPs"的纯扩展（1.6T 参数、7× 提速）证明能力来自 scale，而非 R2 的"冻结基底 + 有界自适应控制器"。**
Switch 把整网（含路由器）端到端联合训练以堆参数，似乎说"把基底做大、端到端训"才是正道，与 VZ"冻结基底、只在有界控制器层适应"对立。
- **裁决：needs-boundary-condition（实为正交，不构成证伪）。** Switch 是**rare-heavy 离线预训练**阶段的扩展技巧（FLOPs/token 恒定下扩参数），与 VZ 的**运行时 online-fast 适应层**不在同一时间尺度；它没有声称"运行时端到端更新基底可行"。
- **边界**：Switch 式扩展只属于"如何造一个更强的**冻结基底**"（rare-heavy artifact 训练），不得据此对运行时基底做在线端到端梯度（R2）→ [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md)。

**反证 C：AutoAugment 消融显示"随机搜索 ≈ RL"，因此 VZ 在 z_t 空间学控制器、上 RL 是过度设计。**
作者明确"RL 只是图方便，随机搜索/进化策略一样好；主要贡献是搜索空间设计而非优化器"。这似乎贬低了"学习控制器"的价值。
- **裁决：needs-boundary-condition（且对 VZ 是正向压力）。** 它证伪的不是"该用学到的控制器"，而是收窄主张为：**控制器的搜索空间/表示设计 > 具体优化算法**。这与 VZ"控制发生在结构化 latent 空间（z_t）"一致——关键是把空间设计好，RL 只是其中一种搜索手段。
- **边界**：VZ 不应迷信"RL 本身"，应把工程投入放在 z_t 控制空间的结构设计上；RL/进化/随机搜索按 ROI 选 → [`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md)。

### 2.3 局部算法借鉴（算法级解耦）

> 关于"Switch 路由是可借鉴的专家切换机制，还是只是扩展手段？"——**精确结论**：路由器本身（内容相关的离散 top-1 选择 + 可微负载均衡）是一个**可借鉴的切换机制**；而"堆到 1.6T 参数"是**不可借鉴的扩展叙事**。两者必须解耦：借机制、弃叙事。

| # | 机制（剥离叙事） | 目标 VZ spec | 落地动作 | 预期收益 | 风险 / 前提 |
|---|---|---|---|---|---|
| 1 | **Switch top-1 路由**：一个学到的路由器按内容对输入做**离散 top-1 选择**，在一组有界子模块间切换，门控值保持可微 | [`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md), [`affordance.md`](../../../docs/specs/affordance.md) | 用学到的路由器在 z_t 空间对**一小组有界控制器/技能专家**做 per-turn top-1 切换（β_t 的工程实现），而非关键词→行为硬编码 | 内容相关、可微、常数算力的行为切换；与 `no-keyword-matching-hacks` 一致；切换是涌现而非手写规则 | **硬切换不稳定**（论文自承）需稳定化；切换必须在 latent 控制空间、不溢出 token 空间（R4）；专家是**有界控制器**而非冻结基底（R2） |
| 2 | **可微负载均衡辅助损失** `α·N·Σ fi·Pi`：惩罚路由坍缩，逼近均匀使用专家 | [`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md), [`credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md) | 给 #1 的 β_t 切换加同型 anti-collapse 正则，避免控制器永远只用一个专家/regime；作为信用分配的辅助项 | 防止"切换单元退化为单一模式"，保持时间抽象的多样性与可用性 | 系数需调（论文 α=10⁻²）；正则目标是"可用性"非"质量"，不得反向变成学习源（R12 只读）；不得让均衡压过主目标 |
| 3 | **AutoAugment 学习式策略搜索**：控制器采样 policy → 固定 child 用**下游验证准确率**当奖励 → 策略梯度更新 → 产出**可迁移的离线 policy 工件** | [`credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md), [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md) | 在 **rare-heavy 离线**阶段，用可验证 eval 当奖励搜索控制器/策略工件，冻结后下发；online 层只消费工件、保持有界；变更经 ModificationGate | 用数据驱动的 policy 替代手调阈值，且可跨场景迁移；天然契合多时间尺度分层 | 搜索奖励**必须是可验证 eval（R12，A-Lab 教训）**，否则放大错误；搜索昂贵；权重级变更必须门控（R9/R10）；RL 非必需（结构 > 优化器，反证 C） |

## 3. 一句话定位
Periodic Labs（Fedus/Cubuk）对 VZ 的**最大价值是一条反向教训而非一项背书**：A-Lab 自主合成结果被质疑，证明"自动化 ≠ 可验证"，正向印证 VZ 的 R12（评估先做硬）与 R-PE（PE 来源不外包）；而两篇创始人奠基工作在解耦叙事后留下两个高质量局部机制——**Switch top-1 路由 + 负载均衡损失**可作为 β_t/affordance 切换的工程实现，**AutoAugment 学习式策略搜索**可作为 rare-heavy 离线学控制器工件的范式（前提是奖励必须是可验证 eval）。

## 附：本地论文清单（同目录 PDF）
- `switch-transformers-trillion-parameter-sparsity (founder, Fedus)-2101.03961.pdf` — Switch Transformers / 稀疏 MoE top-1 路由（2021，创始人 Fedus）
- `autoaugment-learning-augmentation-from-data (founder, Cubuk)-1805.09501.pdf` — AutoAugment / RL 搜索数据增强策略（2018，创始人 Cubuk）
- （付费·本地无 PDF，UNVERIFIED）GNoME — Scaling deep learning for materials discovery，doi:10.1038/s41586-023-06735-9
- （付费·本地无 PDF·结果被质疑）A-Lab — An autonomous laboratory for accelerated synthesis，doi:10.1038/s41586-023-06734-w
