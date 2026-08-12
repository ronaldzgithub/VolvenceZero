# 杨柳（Liu Yang）论文集研究项目：学习理论 × Volvence 四能力轴

> 研究对象：杨柳（Liu Yang），CMU 机器学习博士（2013，导师 Avrim Blum & Jaime Carbonell，博士论文《Mathematical Theories of Interaction with Oracles》），Steve Hanneke 长期合作者（16 篇）。学术轨迹：华中科技大学（图像配准）→ MSU（度量学习/半监督，Rong Jin 组）→ CMU（主动学习/查询复杂度理论）→ IBM Research / Yale（图算法、组合优化）。
> 研究目的：逐篇评估其 40 篇论文对 volvence 项目的意义与结合点（思想级 + 算法级），映射到 Appendable / Readable / Learnable / Steerable 四能力轴。
> 分析基准日：2026-08-12。本项目是**只读研究文档**：不改任何 spec、不动任何 WiringLevel、不构成任何 prereg。

## 一句话结论

这批论文对 volvence 的价值不是可移植的算法库，而是一套**"查询 / 标注 / 验证预算的数理会计学"**——恰好覆盖 volvence 当前最缺理论语言的部位：Gate 4 之后"省标签主张还能怎么立"、C2 人类验证锚"为什么只能是验证锚"（信息论必然，不只是契约选择）、记忆保鲜与 consolidation 节律的定量设计规则。其中**负结果与正结果同等重要**：多篇下界直接构成对外表述的禁语清单。

## 文档结构

| 文件 | 内容 |
|---|---|
| [00_PAPER_INDEX.md](00_PAPER_INDEX.md) | 40 篇总索引：编号 / 发表 / 作者序 / 簇 / 档位 / PDF / 来源；补充材料与不可下载项登记 |
| [01_ACTIVE_LEARNING_LABEL_COMPLEXITY.md](01_ACTIVE_LEARNING_LABEL_COMPLEXITY.md) | 簇 1（10 篇）：主动学习标签复杂度 ↔ Learnable / Gate 4 / 工作流 C |
| [02_RL_APPRENTICESHIP_SELECTIVE_SAMPLING.md](02_RL_APPRENTICESHIP_SELECTIVE_SAMPLING.md) | 簇 2（4 篇）：RL / 学徒学习 / 择时查询 ↔ Steerable gate 择时、C2 送标预算 |
| [03_TRANSFER_PRIOR_NONSTATIONARY.md](03_TRANSFER_PRIOR_NONSTATIONARY.md) | 簇 3（7 篇）：迁移先验 / 漂移 / 非平稳 ↔ Appendable 记忆保鲜、群体先验 |
| [04_TESTING_AUDIT_THEORY.md](04_TESTING_AUDIT_THEORY.md) | 簇 4（3 篇）：性质测试 ↔ Readable 审计预算、预检功效 |
| [05_EARLY_ML_AND_THEORY_MISC.md](05_EARLY_ML_AND_THEORY_MISC.md) | 簇 5（16 篇）：早期 ML/CS 杂项简评 + 学术谱系 |
| [06_VZ_INTEGRATION_FOUR_AXES.md](06_VZ_INTEGRATION_FOUR_AXES.md) | **综合交付**：四轴总图、15 条转化候选（近/中/远期）、8 条不借鉴项、合并禁语清单、主线方案对账 |
| [download-summary.md](download-summary.md) | 下载与 PDF 校验记录 |
| `download_yangliu_papers.sh` | 幂等下载脚本（37 篇可获取论文 + 6 份补充材料，含 PDF 完整性校验） |
| `papers/` | 37 篇正式 PDF + 1 份 EuroCG 论文集 booklet + 6 份补充材料（含博士论文全文）；#02/#37/#40 无公开版，仅引用登记 |

## 档位分布（以簇文档深读判定为准）

- **A 档 16 篇**：直接进入四轴整合清单（理论模板 / 设计定标级）
- **B 档 12 篇**：思想级参照
- **C 档 12 篇**：背景登记（含 3 篇无公开版）

## 最重要的五个结合点（详见 06 §4）

1. **Gate 4 后继 prereg 的理论骨架**（#07/#25/#09）：省标签主张只能挂 0-1 型择时决策目标（凸目标 minimax 判零改善）；对话域处于高噪声 regime，效应量只能承诺多项式因子，power 按 `ν²/ε²·d` 主项核算。
2. **C2 验证锚定位的信息论升格**（#16/#30/#31）：熵下界证明小样本专家标注学不动 policy——"只作验证锚"是数学必然；oracle 选择判据 `c_j/(1/2−α_j)²` 形式化了主线方案 §4.2 的升级条件，可直接做远期升级 prereg 骨架。
3. **记忆保鲜量纲规则**（#12/#06）：有效窗口 `√(d/Δ)`、误差地板 `Ω(√(dΔ))`、相关流有效样本 `m/k`——decay 是第一性设计而非容量妥协，cadence 分层是统计必要而非算力妥协；#12 的自适应扩窗统计量是唯一"信号合法 + 无需漂移率先验"的即插构件。
4. **低维 readout 架构的统一理论辩护**（#15/#21/#10）：先验估计速率随维度崩塌 ⇒ 群体级个人化必须在 16 维 typed readout / rank-3 z_t 空间做；可辨识性定理证明 P2 多 probe 成对设计是必要而非偶然。
5. **审计预算的第一性框架**（#14/#23）：testing ≪ learning 分离 + testing dimension 把"预检零新增 GPU、formal 才烧大预算"升格为可论证纪律；heldout 满分按 rule of three 反推分辨率、预检冻结门槛附功效论证。

## 使用纪律

- 本项目一切"转化候选"落地必须按 `AGENTS.md` §8 另立收敛包并先冻结 prereg。
- 理论迁移 ≠ 证据：Gate 5 / 七日 formal / Gate 4 后继的判词仍须由 prereg run 产出，本项目不改变 scorecard。
- 引用任何定理前先核对假设匹配度（i.i.d. / 无噪 oracle / 显式假设类逐条核对），不满足只做量级/结构参照——各簇文档每篇均附"风险与不适配"一节。
- 对外表述先查 [06 §6 禁语清单](06_VZ_INTEGRATION_FOUR_AXES.md)。
