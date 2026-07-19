# Neo Labs 论文调研 — 实验室名册（bio-heavy, 2026-06）

调研日期：2026-06-13

## 口径

- **范围**：硅谷主流的**新型实验室 / 公司**（"neo labs"），偏 **生物 / 神经科学 / 科学自动化** 方向。
- **排除**：OpenAI、Google DeepMind、Google Research、Anthropic —— 这四家已在
  [`../arxiv-survey-2026-05.md`](../arxiv-survey-2026-05.md)、
  [`../core-author-paper-assessment-2026-05.md`](../core-author-paper-assessment-2026-05.md)、
  [`../deepmind-author-paper-assessment-2026-05.md`](../deepmind-author-paper-assessment-2026-05.md) 中评估。
- **以公司为单位**：每个实验室一个评估单元，找其**主要创作者**（创始人 + 首席科学家）的代表论文。
- **对齐三轴 + 7 primitive**（与 [`../../docs/next_gen_emogpt.md`](../../docs/next_gen_emogpt.md) R-ID 对齐）：
  - R2（冻结基底 vs 自适应控制器）、R3/R4（时间抽象 + latent 控制）、R-PE（预测误差一级信号）、
    R5/R6（记忆连续谱）、R9/R10/R15（有界自修改 + 可回滚）。
- **下载**：PDF 落到 [`../papers/neolabs/<lab>/`](../papers/neolabs/)，与既有 `papers/dm/` 物理隔离。
  Nature/Cell/Science 付费墙的论文以 DOI 引用 + bioRxiv 预印优先下载。
- **深度分析**：每家的 PDF-grounded 深度分析（核心逻辑 + 确证/反证/局部算法三视角）见 `labs/<lab>/analysis.md`（与该 lab PDF 同目录）；跨 lab 反证矩阵与局部算法 ROI 台账见 [`98_deep_analysis_framework.md`](98_deep_analysis_framework.md)；确证导向横向综合见 [`99_synthesis_vz_mapping.md`](99_synthesis_vz_mapping.md)。

## 名册（33 个）

### Group A — 脑启发 / 神经科学认知（对 VZ 最直接相关）

| # | 实验室 | 主要创作者 | 一句话主张 |
|---|---|---|---|
| A1 | Numenta / Thousand Brains Project | Jeff Hawkins, Subutai Ahmad | 千脑理论：智能 = 大量皮层柱的传感运动投票模型 |
| A2 | Liquid AI | Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus | Liquid / 闭式连续时间网络，自适应高效基础模型 |
| A3 | VERSES AI | Karl Friston (首席科学家), Gabriel René | Active Inference / 自由能原理的工程化"数字大脑" |
| A4 | Stanhope AI | Karl Friston, Rosalyn Moran, Biswa Sengupta | 边缘端 active inference，世界模型 + 可解释信念 |
| A5 | Cortical Labs | Brett Kagan, Hon Weng Chong | 生物神经元 + 硅片（DishBrain / wetware）的目标导向学习 |
| A6 | Cartesia | Albert Gu, Karan Goel, Chris Ré (源头) | 状态空间模型（SSM/Mamba）做实时长上下文基础模型 |
| A7 | Symbolica AI | George Morgan | 范畴论 / 结构化推理，替代纯规模化的符号-神经路线 |

### Group B — 自主 AI 科学家 / 闭环发现（映射 R-PE + SSL→RL + 自修改）

| # | 实验室 | 主要创作者 | 一句话主张 |
|---|---|---|---|
| B1 | Future House | Sam Rodriques, Andrew White | 自主 AI 科学家（PaperQA / Aviary / Robin），假设→实验闭环 |
| B2 | Lila Sciences | Geoffrey von Maltzahn, Andy Beam, John Gregoire | "科学超级智能" + 自主机器人实验工厂 |
| B3 | Periodic Labs | Liam Fedus, Ekin Dogus Cubuk | 自主材料/化学实验室，AI agent 设计并执行物理实验 |
| B4 | Recursive Superintelligence | Richard Socher, Jeff Clune, Tim Rocktäschel, Alexey Dosovitskiy, Yuandong Tian | 递归自我改进：把 AI 研究闭环（构思→实现→验证）全自动化，方法论 = open-endedness（血统亦覆盖 Group D 架构） |

### Group C — 生物基础模型 / 虚拟细胞 / 蛋白（生物核心）

| # | 实验室 | 主要创作者 | 一句话主张 |
|---|---|---|---|
| C1 | EvolutionaryScale | Alexander Rives, Tom Sercu, Sal Candido | ESM3：蛋白序列-结构-功能联合生成前沿模型 |
| C2 | Arc Institute | Patrick Hsu, Silvana Konermann, Brian Hie | Evo 基因组基础模型 + 虚拟细胞（State）/ AI 虚拟实验室 |
| C3 | Isomorphic Labs | Demis Hassabis, Max Jaderberg, Sergei Yakneen | AlphaFold3 谱系的药物设计平台（从 DeepMind 拆出） |
| C4 | Chai Discovery | Joshua Meier, Jack Dent | Chai-1/2：开放的生物分子结构预测基础模型 |
| C5 | Profluent Bio | Ali Madani | OpenCRISPR / ProGen：蛋白与基因编辑器的语言模型设计 |
| C6 | Inceptive | Jakob Uszkoreit | RNA "生物软件"：可学习的 mRNA 设计 |
| C7 | Latent Labs | Simon Kohl | 生成式蛋白设计（前 DeepMind AlphaFold 团队） |
| C8 | Recursion | Chris Gibson, Imran Haque | 高通量表型组学 + 虚拟细胞图谱（Phenom 基础模型） |
| C9 | Insitro | Daphne Koller | 机器学习 + 高内涵细胞数据驱动药物发现 |
| C10 | Generate Biomedicines | Gevorg Grigoryan, Molly Gibson | Chroma：蛋白结构扩散生成模型 |
| C11 | Xaira Therapeutics | Marc Tessier-Lavigne, Hetunandan Kamisetty, David Baker (关联) | 大规模生成式蛋白/抗体设计平台 |
| C12 | Noetik | Jacob Rinaldi, Ron Alfa | OCTO：空间多组学虚拟细胞自监督模型 |
| C13 | CZI / Chan Zuckerberg Initiative — Virtual Cell | Theofanis Karaletsos, Stephen Quake | rBio / TranscriptFormer：开放虚拟细胞基础模型 |
| C14 | Basecamp Research | Glen Gowers, Oliver Vince | 全球生物多样性测序 → 蛋白基础模型训练数据底座 |

### Group D — 前沿架构实验室（与认知架构相关）

| # | 实验室 | 主要创作者 | 一句话主张 |
|---|---|---|---|
| D1 | Sakana AI | David Ha, Llion Jones | 自然/进化启发，AI Scientist + 模型合并 + 自我改进 |
| D2 | World Labs | Fei-Fei Li, Justin Johnson, Ben Mildenhall, Christoph Lassner | 空间智能：可生成可交互的 3D 世界模型 |
| D3 | Physical Intelligence (π) | Sergey Levine, Karol Hausman, Chelsea Finn | π0/π0.5：机器人的通用视觉-语言-动作（VLA）基础模型 |
| D4 | Skild AI | Deepak Pathak, Abhinav Gupta | 通用机器人"基础脑"，跨形态泛化（含内禀好奇心血统） |
| D5 | Thinking Machines Lab | Mira Murati, John Schulman, Lilian Weng | 可理解、可协作的"交互模型"，实时多模态人机协同 |
| D6 | Reflection AI | Misha Laskin, Ioannis Antonoglou | 自主编码/通用 agent（前 DeepMind RL 团队） |
| D7 | Ineffable Intelligence | David Silver | superlearner：纯经验 RL + self-play，不预训练/不模仿人类数据，以"经验时代"为北极星 |
| D8 | humans& | Eric Zelikman, Noah Goodman, Andi Peng, Yuchen He, Georges Harik | 以人为中心的前沿实验室：AI 作为"人与人关系的连接组织"，long-horizon/multi-agent RL + memory + 用户理解（对 VZ 关系/EQ 愿景最直接） |

> 名册外标注：**SSI (Safe Superintelligence)** —— Ilya Sutskever 创立，完全 stealth，无公开论文，不计入下载与评估，仅作背景登记。

## 主要创作者血统映射（为何与 VZ 共振）

- **Active Inference 一脉（A3/A4）**：Friston 的自由能/预测编码 = R-PE "预测误差是一级原始信号" 的最深理论母体。
- **千脑 / SSM / liquid（A1/A2/A6）**：参考帧、连续时间动力学、状态空间记忆 = R2 基底 + R5/R6 记忆连续谱的非 Transformer 候选。
- **自主科学家闭环（B1/B2/B3 + D1）**：假设→实验→学习闭环 = SSL→RL 交替 + R10 自修改的可验证目标范例。
- **递归自我改进 / open-endedness（B4）**：Darwin Gödel Machine 的"FM 改写自身代码 + 归档式开放探索 + 经验验证门控 + 沙箱可追溯" = R9/R10/R15 有界自修改 + 可回滚 + 评估证据先行的最直接工程范例。
- **纯经验 RL / 经验时代（D7）**：David Silver 的"Reward is Enough" + AlphaGo Zero self-play = R-PE"奖励/预测误差是一级原始信号"的纯粹形态；其"无预训练"假设与 VZ R2"冻结基底 + 自适应控制器"构成对立对照，可用于校准分层是否成立。
- **以人为中心 / 关系即目标（D8）**：humans& 把"人与人的关系、信任、连接"设为公司级目标，是名册里与 VZ"关系与主体性（EQ + 信任）优先于 IQ"最贴合的一家；Quiet-STaR（token 之上的可学习内部思考）= R3/R4 latent 控制，RSA / Pragmatic Feature Preferences（把用户语用/心智建模注入奖励学习）= R11 `user_model`/`relationship_state` + R-PE→credit readout 的方法论母体。
- **生物基础模型（Group C）**：在非语言模态上的"冻结大基底 + 下游适配"范式，是 R2 跨模态泛化的对照。
- **VLA / world model（D2/D3/D4）**：latent 动作空间 + 想象 rollout = R3/R4 latent 控制器的工程对照面。
