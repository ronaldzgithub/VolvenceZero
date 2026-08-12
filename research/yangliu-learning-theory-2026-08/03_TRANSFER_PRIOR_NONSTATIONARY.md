# 簇 3 · 迁移先验估计 / 漂移 / 非平稳（7 篇逐篇深析）

> 研究项目：`research/yangliu-learning-theory-2026-08/`；总索引见 [00_PAPER_INDEX.md](00_PAPER_INDEX.md)。
> 分析对象：杨柳（Liu Yang，CMU ML PhD 2013，导师 Avrim Blum & Jaime Carbonell；Steve Hanneke 长期合作者）"迁移先验 / 漂移 / 非平稳"簇，共 7 篇：#08/#10（transfer learning 与先验可辨识性）、#15/#21（先验估计 minimax 速率）、#11（drifting distribution 主动学习）、#12（drifting target concept）、#06（nonstationary β-mixing）。
> 伴读材料：博士论文 `papers/supp-phd-thesis-oracles-cmu2013.pdf` 第 7 章（= #08/#10）、第 8 章（= #15 主体）、第 9 章（preference elicitation 扩展）、第 10 章（= #11）、第 11 章（= #12，另含会议版没有的 §11.4.2 Random Drifts）。
> 分析基准日：2026-08-12。所引 Volvence 文档为当日版本：`docs/appendable-readable-learnable-steerable.md`（下称"四轴 charter"）、`docs/moving forward/主线提升方案_2026-08.md`（下称"主线方案"）、`docs/specs/continuum-memory.md`、`docs/specs/personal-conditioning.md`、`docs/specs/multi-timescale-learning.md`。

---

## 0. 簇定位：本簇 ↔ Appendable 轴的三层理论对应

**Appendable 轴主张什么。** 四轴 charter §2.1："系统的内部状态必须能按时间尺度追加，而不是每次对话从空上下文重生"；§2.2 的实现面是 CMS 四层记忆（瞬态/情景/持久/派生，`vz-memory` ACTIVE）、Gate 8 wake/sleep 与 Gate 11 per-user continuity、State-KV / Prefix-KV 个人条件化载体（Personal 辨识证据 pass，Relationship 轨 chance 封存）、9 类语义 owner。§2.3/§8 的诚实边界：Appendable 当前证明的是**可运行、可审计、可回滚**；"多频优于单频的因果净增益"（Gate 5）尚未成立，七日 formal 在 v1 仪器上停跑、N+1 新 prereg 执行中。

**本簇提供什么。** 杨柳这 7 篇恰好把 Appendable 的两个悬而未决的"值不值 / 多快"问题形式化成了可判定的定理族，且三层对应干净：

```text
(a) 跨用户先验层（#08/#10/#15/#21）
    每个用户 = 一个 target concept h*；用户群 = 未知先验 π 下的 i.i.d. 采样。
    定理回答：π 能否从每用户有界数据中估出（能，d 个标签/任务即可辨识）、
    估多快（minimax 速率显式依赖 T、d、光滑度 α）、估出后新用户的
    适应成本降到什么水平（趋近"先验已知"的 SC，且自验证成本 o(1/ε)）。
    ↔ Appendable 的"为什么值得追加"：个人化的群体闭环有可证明收益。

(b) 单用户漂移层（#11/#12）
    用户随时间变化 = drifting target concept（偏好/状态在变）
    或 drifting distribution（话题/情境分布在变，身份稳定）。
    定理回答：漂移率 Δ 与可达误差的定量关系（误差地板 √(dΔ)、
    有效记忆窗口 √(d/Δ)、次线性错误 ⇔ ΣΔ_t=o(T)、跳变代价 √(d/Δ)）。
    ↔ Appendable 的"多快必须刷新"：记忆/载体的保鲜速率与衰减律。

(c) 非 i.i.d. 对话流层（#06）
    对话流时间相关 = 非平稳 β-mixing 过程。
    定理回答：相关数据上还能不能学（能，只要 β_k=O(k^{-r}) 且
    累计漂移 O(T^α)、α<1）、多快（累计超额风险次线性，指数显式）、
    怎么学（跳采样 + 滑动窗 ERM——有效样本量是 m/k 而非 m）。
    ↔ Appendable 的 cadence 分层：session-medium/background-slow
    的间隔化更新不只是省算力，是统计上必要的去相关。
```

必须先说清楚这套理论回答的**是什么、不是什么**：它回答的是 Appendable 的"为什么值得追加"（先验可估 → 群体个人化有可证明收益）与"多快必须刷新"（漂移率 → 保鲜窗口/衰减率的定量设计规则），**不是**工程实现本身，更不是 Gate 5 / 七日 formal 的替代证据。主线方案 §1.1 scorecard 的"多尺度记忆"行——机制证据 = Gate 5/8/11 已有结论，系统缺口 = 七日 formal 16/36 停跑与 MSC formal 三重阻断——不因本簇任何定理而改变一个字；本簇给的是设计定标（效应量该有多大、窗口该多长、预算该多少）与假设检验框架（哪些主张理论上不可能，prereg 不许承诺）。

## 0.1 分析纪律声明

1. **理论迁移 ≠ 证据**。Gate 5 多频净增益、七日 formal 判词仍需按主线方案 §2.1（A1 双预检：§0 不变量 7 分辨力预检 + 不变量 8 臂→目标传导预检）的 prereg run 产出；本簇一切"支撑"措辞只指"设计视角与假设检验框架"，不指判词。本簇定理全部依赖 i.i.d.（#08/#10/#15/#21）、独立性（#11/#12）或显式放松后的 mixing 假设（#06），引用时逐条核对，不满足的只做量级/结构参照。
2. **evaluation/judge 禁止作为学习信号**（R12，主线方案 §4.0）。本簇转写候选涉及的一切"漂移检测量 / 窗口选择统计量 / 先验估计输入"，信号来源只允许两类合法项：PE 通道（N+1 表示 PE 与其派生的 mismatch 计数——#12 Theorem 1 的窗口统计量恰好是"窗口内最优拟合错误数"，属 PE 家族）与 owner 发布的 typed readout（快照指纹、16 维坐标等）。七日 continuity 七项、companion-bench judge 分数在任何转写中不出现。
3. **个人化学习只在控制器层/载体层**。per-user prior、漂移自适应窗口、consolidation 调度的一切落点都在 State-KV / CMS / conditioning owner 的有界参数上（`personal-conditioning.md` §5 的"共享部分是一个小型 profile encoder / projector；共享投影的新版本必须经过 `ModificationGate`"；`continuum-memory.md` 的 `promotion_threshold` 属"可回滚低风险自适应参数"）——不改冻结基底，不做 token 空间 RL。
4. **C2 人类标注只作验证锚**（`steering-human-anchor.md` 契约，主线方案 §4.2）。本簇不产生任何以人类标注为输入的估计器或漂移检测器；涉及"标注预算"的结合点只落在验证锚自身的功效核算。
5. **负结果与正结果同权**。#12 的 ΣΔ_t=o(T) 充要条件与 Ω(√(dΔ)) 误差地板、#15 的 minimax 下界、#11 的 Ω(T^{m/(m+1)}) 下界，都是"关系连续性 / 持续适应"对外措辞的诚实边界素材，在簇级小结显式列出。

---

### 08. A Theory of Transfer Learning with Applications to Active Learning（Machine Learning 2013）

**文件**：`papers/08-theory-transfer-learning-active-ml2013.pdf`（28 页；深读 §1–§5 全部主结果与证明骨架）

1. **基本信息**：Liu Yang, Steve Hanneke, Jaime Carbonell（**杨柳一作**，打破字母序）；Machine Learning 90(2), 2013。贡献声明："transfer learning setting in which a finite sequence of target concepts are sampled independently with an unknown distribution from a known family ... derived the asymptotic label complexity bound in order to learn all targets to a specified expected accuracy."
2. **问题设定与核心结果**：VC 类 C（维 d），固定数据分布 D，度量 `ρ(h,g)=D(x:h(x)≠g(x))`；先验族 `{π_θ:θ∈Θ}` 在总变差距离下 totally bounded，真参数 θ\* 未知；T 个任务的目标 `h*_t ~ π_θ*` i.i.d.，每任务数据 `Z_t(θ*)={(X_ti, h*_t(X_ti))}`。算法须对**每个**任务保证 `E[ρ(ĥ_t, h*_t)] ≤ ε`，度量指标是 T 任务平均的期望标签请求数。**Theorem 1**：存在估计器 `θ̂_T = θ̂_T(Z_1d,…,Z_Td)`——每任务只用**前 d 个标签**——与可计算函数 R、δ 使 `P(‖π_θ̂ − π_θ*‖ > R(T,α)) ≤ δ(T,α) ≤ α`，且 `R,δ → 0`（对 θ\* 一致，R 可在算法内计算）。**Theorem 2**：d 是最小的——存在 D 与 π_1≠π_2 使任意 k<d 个点的联合分布完全相同（构造：均匀先验支撑在 shatter 集的全部 2^d 种标注 vs 支撑在带 parity 约束 `h(x_d)=∏_{j<d}h(x_j)` 的 2^{d−1} 子类，k≤d−1 时边际同为均匀）。**Algorithm A_τ + Theorem 3**：`limsup_T E[S_T(ε)]/T ≤ SC(A_a, ε/4, D, π_θ*) + d`——长程平均标签复杂度收敛到"先验已知"水平，每任务额外成本至多 d 个标签（且 +d 与 ε/4 均可用标准技巧消去/收紧到 (1−o(1))ε）。**Theorem 4（§5）**：任意 VC 类存在 prior-dependent **自验证**主动学习算法使 `SC(A_a,ε,D,π) = o(1/ε)`；对照：prior-independent 自验证 AL 在 interval 类上有 Ω(1/ε) 下界（Balcan–Hanneke–Vaughan 2010），先验相关被动学习是 Θ(1/ε)（Haussler–Kearns–Schapire）。结论：迁移学习不只改常数，能**改进 ε 的渐近依赖**，并弥合"学得快 vs 能验证自己学好了"的差距。
3. **核心机制**：三级归约链。Lemma 1：无限数据序列的分布距离 = 先验距离，`‖π_θ−π_θ'‖ = ‖P_{Z_t(θ)}−P_{Z_t(θ')}‖`；Lemma 3：k 点联合分布距离被 d 点距离控制，`‖P_{Z_tk(θ)}−P_{Z_tk(θ')}‖ ≤ 4·2^{2k+d}k^d·√‖P_{Z_td(θ)}−P_{Z_td(θ')}‖`（归纳杠杆：VC 维 d 保证任意 k>d 个点存在无法实现的标注 ȳ，概率差沿"翻一位靠近 ȳ"的二叉树递归塌到 d 点边际）；Lemma 4：totally bounded 分布族上 minimum-distance skeleton 估计器一致收敛。算法 A_τ：每任务先请求前 d 个标签更新 θ̂；若 `R(t−1,ε/2) > ε/8`（先验估计还不准）则回退被动学习（`m_ε = (16d/ε)ln(24/ε)` 个标签的 consistency ERM）；否则在 TV 球 `B(θ̂,R)` 内选 `SC` 近极小的 θ̌，以 `π_θ̌` 跑 prior-dependent 主动算法 A_a。Theorem 4 的证明是把 Hanneke 2009 的 budget-based（非自验证）算法逐目标 `o(1/n)` 风险经对角化 Lemma 汇聚成先验平均 `o(1/n)`，先验的作用只在**算出停止预算** n_ε——"学习不贵，验证才贵；先验补验证那一半"。
4. **思想结合点**：这是 Appendable"为什么值得追加"的存在性定理。映射：每个用户 = target concept，用户群 = π 下的采样；`personal-conditioning.md` §5 的**群体闭环**（"去标识经验 → 反事实/对比训练 → gate 验证 → 共享投影版本升级"）正是 A_τ 的工程对应——从历史用户的有界证据估计群体结构，用它加速新用户。三个结构级对应值得写实：(i) **每任务 d 个额外标签的上界** ↔ Appendable 主张的"经历按时间尺度写入"成本模型：群体学习不要求任何单个用户付出超过自身所需的证据量，理论上限是"有效维度"个样本——这为"隐私/成本约束下群体先验仍可估"提供论证；(ii) **`R > ε/8` 时回退被动** ↔ `personal-conditioning.md` §2 的 cold-start 契约（缺少个人证据时发布全零快照、"不能凭默认人格猜测用户"）——A_τ 的回退臂就是 cold-start 路径的理论版本：先验估计未收敛时用保守通用行为，收敛后才切个人化；(iii) **自验证 o(1/ε)** ↔ Gate 11 per-user continuity 想回答的"系统能否声明已认识这个用户"：定理说这种自我声明的证据成本在有群体先验时是 o(1/ε)，没有时可能 Ω(1/ε)——群体先验的价值一半在加速学习，一半在**降低验证成本**，后者恰是 volvence 证据体系最贵的部分（七日 formal 的预算之痛，主线方案 §6）。
5. **算法结合点**：可转写对象 = **per-user prior 的表示与估计器**。现实落点不是原始表示空间（见第 7 点），而是 owner 已发布的低维 typed readout 空间：把"用户群先验"定义为 16 维 Personal 坐标（或 State-KV 生成器的低秩输入）上的分布族，由跨用户去标识快照估计，用作新用户 State-KV 的先验初始化 / `PersonalConditioningModule` 的 cold-start→warm-start 过渡。信号来源核对：输入全部是 owner typed readout（16 维坐标、快照指纹），无 judge、无 continuity 分数，合法；估计器本身属 rare-heavy 群体闭环，产物（共享先验参数）经 `ModificationGate` 进入，符合纪律 3。假设匹配度核对：**known totally bounded family**——对 16 维 `[0,1]` 有界坐标可显式构造（参数化密度族），基本满足；**任务间独立**——用户间近似独立可辩护，但同一助手版本对所有用户的耦合是残余风险；**共享固定 D**——不满足（每用户的语境分布不同；本设定假设所有任务共享 X 分布、只有标签函数不同），这是与 #11 的漂移设定互补的缺口；**realizable 无噪声标签**——不满足（用户状态观测有噪）；**任务内 h\* 固定**——长程上不满足（归 #12）。结论：结构与成本模型可转写，数值保证不可引用。
6. **成熟度与档位**：**A**——本簇的奠基性论文，"群体先验 + 每用户有界证据 + 先验补验证"三件套是 Appendable 群体闭环叙事的理论骨架；进入四轴整合清单的方式是设计定标与叙事支撑，非代码直译。
7. **风险与不适配**：Lemma 3 的常数 `4·2^{2k+d}k^d` 随 d 指数爆炸，原始 LLM 表示空间（d 天文数字）上此路不通——必须先压缩到低维 readout（这与 #15 的速率分析共同构成"低维 typed readout 架构"的理论辩护）；渐近结果（R、δ 无显式速率——速率是 #15/#21 的主题）；A_τ 需要对候选先验计算 `SC(A_a,·)`（不可计算量，工程上只能用代理排序）；"known family"在我们这里是建模选择而非事实，族选错时估计器一致收敛到错误对象且无警报——需要配 held-out 拟合优度检查。

---

### 10. Identifiability of Priors from Bounded Sample Sizes with Applications to Transfer Learning（COLT 2011）

**文件**：`papers/10-identifiability-priors-transfer-colt2011.pdf`（13 页；#08 的会议版/前身，深读全文并与 #08 逐节比对）

1. **基本信息**：Liu Yang, Steve Hanneke, Jaime Carbonell（**杨柳一作**）；COLT 2011。贡献声明："I invented the topic..."——她自认的开题之作。与 #08 的关系：设定、Theorem 1（先验一致估计）、Algorithm A_τ 与 Theorem 8（= #08 Theorem 3，`limsup E[S_T]/T ≤ SC(A_a,ε/4,D,π_θ*) + d`）逐字对应；差异在 §4.1 只**引用** Yang–Hanneke–Carbonell 2010 技术报告（后成 #24，AISTATS 2011）的 `o(1/ε)` 自验证结果得到推论 `limsup E[S_T]/T = o(1/ε)`，没有 #08 §5 的完整自验证分析。本条目按任务书要求聚焦增量：**有界样本量下先验可辨识性的论证本身**。
2. **问题设定与核心结果**：核心命题拆成三件。**可辨识性（Corollary，#08 中为 Corollary 1）**：`P_{Z_d(1)} = P_{Z_d(2)} ⇒ π_1 = π_2`——先验被"d 个随机标注点的联合分布"唯一决定，其中 d 恰是 VC 维；这不需要 totally bounded 假设，是无条件成立的结构事实。**最小性（Theorem 2）**：对每个 VC 类都存在 D 与先验对使 k = d−1 个点的联合分布不可区分——parity 构造把"身份信息"藏在恰好第 d 阶的联合统计量里，任何低阶边际全部相同。**热身例**：半开区间类（d=2）上，`π_1`（一半概率全正区间、一半空区间）与 `π_2`（一半 [0,½)、一半 [½,1)）在单点分布 `P_{Z_t1}` 上完全相同，但两点条件概率 `P((+1,+1)|(1/4,3/4))` 即可区分——"一个探针问不出身份，两个联合探针可以"。
3. **核心机制**：同 #08 的 Lemma 链（无限序列距离 = 先验距离 → k 点距离被 d 点距离平方根控制 → skeleton 估计）。会议版的叙述重心在"为什么标准密度估计不适用"：我们从未观测 h\* 本身，只观测每任务有限个标签——可辨识性必须穿过这层间接观测建立，这是与 Baxter 1997（full Bayesian、已知超先验、只改常数）的本质区别；本文的 empirical Bayes 设定把 θ\* 当常数估计，结论是渐近达到"先验已知"性能而非常数改善。
4. **思想结合点**：可辨识性定理与 Appendable 证据线上**已经 pass 的那条**——Personal State-KV 盲判归属——是同一个问题的两面。四轴 charter §8：Appendable 已证明项包括"Personal State-KV pass（盲判归属）"；`personal-conditioning.md` §3.4 的 P2 held-out pairwise 识别（两组 P2 合计 G-prefix 56/64、accuracy 0.875，A-pure 控制覆盖随机）实证了"16 维状态经 Prefix-KV 进入冻结基底后，行为分布足以区分用户状态"。本篇给这条证据线的理论定位是：**行为分布对身份的可辨识性存在一个由假设类复杂度决定的最小联合观察窗**——单探针边际可以完全不含身份信息（Theorem 2 的 parity 构造是极端版），身份藏在多探针的**联合**统计量里。这解释了为什么 P2/P3 证据设计必须是多 probe 套件 + 成对比较（而非单 probe 准确率），也为"多少个 probe 才够"给出结构答案：至少要达到用户状态族的有效 VC 维量级。
5. **算法结合点**：可转写对象 = **盲判归属类证据（P2/P3、Gate 11 行为探针）的 probe 数下限核算**。做法：把"用户状态族"取为 16 维 readout 经冻结投影后的行为函数族，其有效维度（≤16 的量级）给出 probe 套件的最小规模参照——现行 16 probes 的设计恰在该量级；若未来读出维度扩容（P5-d D0 门控的 full-dim basis 包），probe 套件必须同步扩容，否则识别失败可能是"观察窗小于 d"的构造性问题而非"载体无效"。信号来源核对：probe 行为分布 + 盲评 matching 属既有证据面，无新信号。假设匹配度核对：**i.i.d. probe 点**（probe 是设计出来的固定集，非 i.i.d.——定理的随机点假设不满足，但固定设计点只会更有利，方向安全）；**无噪声标签**（生成温度 0 时近似成立）；**先验族已知**（用于识别的只是可辨识性方向，不依赖此条）。不可转写部分：Theorem 1 的估计器本体（skeleton 估计在行为分布空间不可计算）。
6. **成熟度与档位**：**A**（与 #08 合并计价：#08 承载算法与自验证结果，本篇承载可辨识性论证与"最小观察窗"教训；总索引两篇均 A）。
7. **风险与不适配**：与 #08 共享全部假设缺口；另有一条本篇特有的解释风险——可辨识性是**群体先验**层面的（π 可辨识），P2 检验的是**单用户状态**层面的可区分性，两者相关但不等价：前者允许两个用户个体不可区分只要群体分布可辨识。引用时必须区分"π 可估"与"个体可识别"，否则会把理论支撑安错位置。

---

### 15. Bounds on the Minimax Rate for Estimating a Prior over a VC Class from Independent Learning Tasks（ALT 2015）

**文件**：`papers/15-prior-estimation-vc-minimax-alt2015.pdf`（23 页；深读 §1–§5 全部定理与证明主体）

1. **基本信息**：Liu Yang, Steve Hanneke, Jaime Carbonell（**杨柳一作**）；ALT 2015。贡献声明："proving the optimal rates of convergence for estimating a prior distribution over a VC class from a sequence of independent data sets labeled by independent target functions sampled from the prior."
2. **问题设定与核心结果**：#08 的设定加两个量化元素：先验密度 `f_θ = dπ_θ/dπ_0`（相对参考测度 π_0）满足 `(L,α)-Hölder` 光滑（`|f(h)−f(g)| ≤ L·ρ(h,g)^α`，α∈(0,1]），风险取 `sup_θ* E‖π_θ̂ − π_θ*‖`（TV）。**Theorem 1（上界，主结果）**：每任务 d 个样本时存在估计器使 `sup_θ* E‖π_θ̂T − π_θ*‖ = Õ(L·T^{−α²/(2(d+2α)(α+2(d+1)))})`。**Theorem 2（下界）**：存在 VC 维 d 的构造使任意估计器 `≥ C(d,L,α)·T^{−α/(2(d+α))}`。**Theorem 3（全数据参照）**：若允许每任务观测完整 `Z_t(θ*)`，下界回到经典密度估计率 `Ω(T^{−α/(d+2α)})`。三个指数的排序把"每任务只看 d 个样本的代价"夹出来了：全数据 `α/(d+2α)` ≫ 有界样本下界 `α/(2(d+α))` ≫ 有界样本上界 `α²/(2(d+2α)(α+2(d+1)))`（上下界不匹配，紧性 open）。定性结论：**先验可估但速率对 d 灾难性地慢**——α=1 时上界指数约 `1/(4d²)` 量级，d 稍大 T 就要天文数字。
3. **核心机制**：上界四步。(i) `k = O((d/γ)log(1/γ))` 个随机点以概率 1−γ 把 C 划分成 L1(D) 直径 < γ 的 cell（PAC 划分）；(ii) 光滑性 ⇒ 先验 TV 被"k 点标注联合分布 TV + Lγ^α"控制：`‖π_θ−π_θ'‖ < Lγ^α + ‖P_{Y_k|X_k}(θ) − P_{Y_k|X_k}(θ')‖`；(iii) 二叉树归纳（#08 Lemma 3 的显式化）：k 点条件分布差 ≤ `(k−d)2^d`·（d 点最大差），配 Sauer 引理 `(ek)^d` 得 `‖π_θ−π_θ'‖ < (L+1)γ^α + 4(2ek)^{2d+2}√‖P_{Z_d(θ)}−P_{Z_d(θ')}‖`；(iv) `{P_{Z_d(θ)}}` 的 ε-覆盖数被先验族覆盖数控制——光滑密度族按 cell 常值 + ε 网格离散化得 `N(ε) ≤ (1/ε)^{O((L/ε)^{d/α})}`，minimum-distance skeleton 估计给 `ε = O(L(log(TL)/T)^{α/(d+2α)})`；对 γ 优化收尾。下界是干净的信息论归约：`X={1..m}`，C = "至多 d 个正点"的类，先验族把 `(m choose d)` 个独立比特 b_i 藏进"支撑 ⊆ X_i 条件下正点个数的 parity"，γ_m = (L/2)m^{−α} 保证 Hölder；关键计数：一个任务只有当它的 d 个样本点**恰好等于** X_i 时才携带关于 b_i 的信息（`E[N_i] ≤ d^{2d}(2γ_m/L)^{2d/α}T`），归约到 `(m choose d)` 个 Bernoulli 偏置判别，用 `P(误判) ≥ (1/32)exp(−128γ²n/3)` 收尾。
4. **思想结合点**：本篇是 Appendable 群体闭环的**功效核算层**，它把 #08 的"能估"变成"要多少用户才估得动"，并给出一条对 volvence 架构最有分量的推论：**群体级个人化学习的可行性随表征维度以 `T^{−Θ(α²/d²)}` 崩塌**——这为 volvence 一系列"压到低维再学"的设计提供了统一的理论辩护：`personal-conditioning.md` §2 的 16 维有界坐标（而非 896/3584 维残差）、rank-3 `z_t` 执行器与 P5-d D0 门控（"full-minus-rank3 无增量证据则保留 rank-3"）、`relationship-conditioning.v2` 的 14 维 dyad 读出。在这些 d≤16 的空间里 Theorem 1 的速率是可用的；在原始残差空间里是数学上的死刑。第二条推论指向证据预算：Theorem 3 vs Theorem 1 的指数差说明**每用户多攒数据对群体估计有多项式级加速**（从 `α²/(2(d+2α)(α+2(d+1)))` 向 `α/(d+2α)` 移动）——Gate 11 的纵向 per-user 证据与群体先验估计不是两笔独立预算，前者直接改善后者的速率档位。
5. **算法结合点**：可转写对象 = **群体先验估计的 power 分析与族设计**。具体：若未来立"cross-user prior 加速新用户 State-KV 初始化"的收敛包，prereg 的样本量核算按本篇写：用户数 T、readout 维 d、假设的密度光滑度 α 代入 Theorem 1/2 的上下界给出可检出效应的区间，避免拍脑袋承诺"N 个用户就能学出人群结构"。族设计侧：光滑性条件的度量是行为度量 `ρ(h,g)=D(h≠g)`——对应我们的"两个用户状态的行为差异概率"，光滑先验 = 人群密度不在行为空间里跳变，这是可以在已有跨用户快照上做只读检查的假设（类比主线方案 §0 不变量 7 的分辨力预检：先验估计包上马前，先预检人群密度的光滑度/覆盖结构）。信号来源核对：全部输入为去标识 typed readout，合法。假设匹配度核对：任务独立（近似成立）、每任务恰 d 个样本（我们每用户样本数不齐——理论按最少者取，保守方向安全）、Hölder 光滑（未验证，需预检）、TV 风险（比我们关心的下游决策质量强，TV 收敛是充分非必要——估计不动 TV 不等于个人化无收益）。
6. **成熟度与档位**：**A**——"低维 readout 架构的理论辩护 + 群体闭环的功效核算模板"两个用途都直接进整合清单；速率本身作定标不作保证。
7. **风险与不适配**：上下界不匹配（`α/(2(d+α))` vs `α²/(2(d+2α)(α+2(d+1)))`），真实速率未定，功效核算只能给区间；skeleton 估计器计算上不可行（#21 的 MLE remark 部分缓解）；下界构造是精心设计的最坏情形，实际人群密度可能远好于最坏——核算按下界乐观、上界悲观双报；常数 `4(2ek)^{2d+2}` 在 d=16 时仍巨大，有限 T 下渐近式失真。

---

### 21. Bounds on the Minimax Rate for Estimating a Prior over a VC Class from Independent Learning Tasks（TCS 2018，#15 期刊版）

**文件**：`papers/21-prior-estimation-vc-minimax-tcs2018.pdf`（32 页；与 #15 逐节比对，本条目只写增量）

1. **基本信息**：Liu Yang（此时署名 Yale）, Steve Hanneke, Jaime Carbonell（**杨柳一作**）；Theoretical Computer Science 716 (2018)。贡献声明（相对 #15 的增量）："bounds on the optimal rates under a smoothness condition on the correct prior"——期刊版补全了证明细节并新增一节关键结果。
2. **问题设定与核心结果（增量）**：主定理 Theorem 1/2/3 与 #15 相同（编号一致）。**新增 §5 "Using More Than d Samples Per Task" 的 Theorem 4**：允许估计器使用每任务**完整**数据集 `Z_t(θ*)` 时，存在估计器达到 `sup_θ* E‖π_θ̂ − π_θ*‖ = Õ(T^{−α/(d+2α)})`——与 Theorem 3 的全数据下界对上（至 log 因子）。这把问题的版图钉死了一半：**无界样本/任务情形已闭合，速率恰为经典密度估计率 `Θ̃(T^{−α/(d+2α)})`**；仍然 open 的只剩有界样本（d 个/任务）情形的紧速率。证明极短：Lemma 1（#08）给 `‖π_θ−π_θ'‖ = ‖P_{Z_t(θ)}−P_{Z_t(θ')}‖`，直接在无限序列分布族上跑 skeleton 估计，覆盖数由先验族覆盖数继承。**其余增量**：(i) 显式的 skeleton 有限样本界 **Lemma 1**：`E‖μ̂_ε − μ*‖ ≤ 3ε + 2√(ln|M_ε|/n)`——可直接用于功效核算的工作公式（`T ≥ (64/ε²)ln N(ε/4)`）；(ii) **实例节**：齐次线性分类器 `h_w(x)=sign(w·x)`、D = 球面均匀，`ρ(h_w,h_w') = (1/π)cos⁻¹(w·w')`，权向量上的 Hölder 密度（含条件良好的投影正态族）诱导概念空间上的 Hölder 先验——光滑性假设在"方向参数化"的假设类上是自然的；(iii) remark：满足条件的族可用 MLE 替代 skeleton 估计（缓解可计算性）；(iv) Theorem 5/6（实值 VC subgraph 类一致估计 + preference elicitation 应用）与 #15 的 Theorem 4/5 相同。
3. **核心机制（增量）**：Theorem 4 的机制价值在于对比——同一个 skeleton 估计器，喂 d 点分布得 `T^{−α/(d+2α)}` 的**平方再打折**（Lemma 3 的 √ 与 `(2ek)^{2d+2}` 因子所致），喂全序列分布直接得 `T^{−α/(d+2α)}`：**有界观察窗的代价全部来自"从 d 点边际重建先验距离"的信息损失**，即 #08 Lemma 3 那一步的平方根。
4. **思想结合点**：增量部分对 Appendable 的意义集中在两处。第一，**"每用户证据深度"的边际价值有了闭式两端**：每用户只留 d 个样本 → 指数 `α²/(2(d+2α)(α+2(d+1)))`；每用户完整轨迹 → 指数 `α/(d+2α)`。这直接对话 `continuum-memory.md` 的容量/衰减设计：CMS 对单用户保留多深的情景层（衰减前的有效样本量），不只影响该用户的个人化质量，还在群体闭环的速率档位之间移动——"衰减策略"是群体学习速率的参数。第二，线性分类器实例把光滑先验落到了"方向参数化"几何上，而 volvence 的载体恰是这个几何：`personal-conditioning.md` §3.3 的 projector artifact（L2 归一 basis 行）、§3.4 Prefix-KV 的低秩生成器、steering reader/executor 的线性方向——"人群密度在冻结表示的方向空间上光滑"是有几何意义、可预检的假设，不是凑数条件。
5. **算法结合点**：可转写对象 = **群体先验包 prereg 的工作公式**。用 Lemma 1 的 `3ε + 2√(ln|M_ε|/n)` 与 `T ≥ (64/ε²)ln N(ε/4)` 做样本量表：给定目标 TV 精度 ε 与假设的族覆盖数（由 16 维网格 + 光滑度算出），解出所需用户数 T——这是比 #15 渐近式好用得多的有限样本工具。信号来源与假设核对同 #15；额外一条：MLE remark 使"可计算估计器"从障碍降为工程问题（16 维参数族上 MLE 可行）。
6. **成熟度与档位**：**A**（与 #15 合并深读、分开计价：#15 承载速率图景与下界构造，本篇承载全数据闭合 + 有限样本公式 + 几何实例三件实用增量）。
7. **风险与不适配**：有界样本情形的紧性仍 open（引用时不得把上界当最优）；Theorem 4 要求"完整轨迹"在我们这里意味着无限对话历史——实际是介于 d 与 ∞ 之间的中间情形，两端插值无理论；线性实例的几何与真实冻结 LLM 残差几何的对应是类比级别（残差流不是球面均匀）。

---

### 11. Active Learning with a Drifting Distribution（NIPS 2011）

**文件**：`papers/11-active-learning-drifting-distribution-nips2011.pdf`（14 页含补充；深读正文全部定理与 Theorem 2/3 证明）

1. **基本信息**：**Liu Yang 唯一作者**；NIPS 2011。贡献声明："stream-based setting allowing the distribution of the examples to change over time. I derived upper bounds on the number of prediction mistakes and number of label requests for disagreement-based active learning algorithms as well as the minimax lower bounds."
2. **问题设定与核心结果**：流式 selective sampling：每步先预测 `Ŷ_t` 再可选请求真标签；边际分布 `D_t` 在 totally bounded 分布族 D 内**任意**游走（无逐步步长限制），目标概念 h\*（或噪声条件分布 η）固定不变。指标：期望错误数 `M̄_T`（噪声情形为超额 `M̄_T − M*_T`）与期望查询数 `Q̄_T`，追求双次线性。**Realizable**：Theorem 1（|D|=1）：`M̄_T = O(d log T)`、`Q̄_T = O(θ_P(ε_T)·d log²T)`；Theorem 2：D totally bounded ⇒ CAL 达 `M̄_T = o(T)`，且 `θ_D(ε)=o(1/ε)` ⇒ `Q̄_T = o(T)`；Theorem 3（覆盖数多项式 `|D_ε| ≤ c·ε^{−m}`）：`M̄_T = O(d^{1/(m+1)}·T^{m/(m+1)}·log²T)`、`Q̄_T = O(θ_D(ε_T)·d^{1/(m+1)}·T^{m/(m+1)}·log²T)`，`ε_T=(d/T)^{1/(m+1)}`；Theorem 4（下界）：含无限二叉树结构的类上 `M̄_T = Ω(T^{m/(m+1)})`，且错误数达到该阶时 `Q̄_T = Ω(T^{m/(m+1)})`——Theorem 3 在 T 依赖上 minimax 紧。**Tsybakov 噪声**（Assumption 5，全族统一 c、α）：ACAL（epoch 重置的 agnostic CAL）达 `M̄_T − M*_T = Õ(T^{((α+2)m+1)/((α+2)(m+1))})`、`Q̄_T = Õ(θ_D(ε_T)·T^{((α+2)(m+1)−α)/((α+2)(m+1))})`（Corollary 2）；Theorem 9 下界 `M̄_T − M*_T = Ω(T^{(1+mα)/(α+2+mα)})`，上下界有 gap。**§6 querying-before-predicting**：允许先查后答时，realizable 情形 LAC 错误数恒为 0（查询数不变）；噪声情形 ALAC 给出 mistakes ≤ `Σ_i δ_i 2^i` 与查询数的显式权衡旋钮（置信序列 δ_i）。
3. **核心机制**：模型选择本身是第一贡献：Bartlett 系的"逐步漂移 ≤ γ"模型允许对手把概率质量持续挤进算法不确定的角落导致线性错误，本文改为**约束分布所在族**（totally bounded），换来次线性。上界证明骨架：取 D 的 ε-覆盖 `{P_1..P_|D_ε|}`，把时间轴按 `k(t)=argmin_k‖P_k−D_t‖` 分桶，每桶攒 `L(ε)=Θ((d/√ε)log(1/√ε))` 个样本后版本空间直径 `≤ √ε + L(ε)ε`（跨桶漂移代价 ≤ ε/步）；CAL 不变量（版本空间恒含 h\*，非分歧点的推断标签免费且正确）保证 `Q_t = Z_t`；查询率由 `D_t(DIS(C[Z_{t−1}]))` ≤ θ_D(r)·max{diam, r} 控制。ACAL 加 epoch 重置（t 为 2 的幂时清空）与局部 Rademacher 阈值 `Ê_t`，把 agnostic 界拼到漂移桶上。
4. **思想结合点**：本篇与 #12 合起来给 Appendable"多快必须刷新"提供了**二分法**：漂移在环境侧（话题/情境分布变、身份稳定）vs 漂移在目标侧（偏好/状态本身变）。本篇是前者——此时**记忆全部保留仍然有效**（CAL 的 Q_t 单调增长、永不丢弃），要补的只是新分布区域的覆盖采样。映射到 CMS：`continuum-memory.md` 的 promotion（持久语义层沉淀"跨情境不变的用户结构"）对应环境漂移下不变的 h\*；decay 只该由目标侧漂移（#12）驱动——**按漂移类型选择保留/衰减**是两篇合并后的设计规则。第二个结合点是 R14 regime 的理论化：totally bounded 分布族 + 覆盖数指数 m，就是"用户语境在有限 regime 空间内游走"的形式化，m 是 regime 空间的有效维度，可达速率 `T^{m/(m+1)}` 随 m 优雅退化——**保持 regime 表示低维有界**（regime owner 的 typed identity、而非开放文本标签）直接改善持续学习的可达速率档位。第三，§6 的先查后答（realizable 零错误）是"先检索记忆/先追问再作答"路径的理论原型：Gate 8 wake/sleep 水合后再进 turn、`MemoryModule` 的 turn-time retrieval 先行，结构上都是"把预测推迟到不确定性消解之后"。
5. **算法结合点**：可转写对象 = **漂移流上昂贵 oracle 的择时调度**。CAL/ACAL 的"只在分歧区请求标签"骨架，在 volvence 的合法信号面上对应：对 N+1 matched settlement（MPS 昂贵）与 C2 验证锚 unit 的选择，优先投向"当前 per-user 模型与保守基线预测分歧"的 episode/时段——这是簇 1（#07/#09）同款结论在**漂移流**上的延拓：本篇证明分歧驱动调度在分布漂移下仍保持查询次线性（只要 regime 族有界）。信号来源核对：分歧信号 = 模型间预测不一致，属 PE/owner readout 家族，合法。假设匹配度核对：**样本独立**——不满足（对话流相关，归 #06 修正）；**目标固定**——分段成立（用户身份在 session 尺度稳定）；**θ_D 有界**——未知，需在低维 readout 族上估计；**Tsybakov 全族统一参数**——最强的一条，实际各 regime 噪声异质，只能取最坏 regime 保守化。
6. **成熟度与档位**：**A**——"约束族而非约束步长"的建模教训 + regime 维度 m 定速率 + 环境/目标漂移二分，三条都进整合清单；她唯一的单作者顶会论文，也是理解其漂移系工作的入口。
7. **风险与不适配**：噪声情形上下界有 gap（引用效应量时用下界保守）；θ_D 取族上 sup，最坏 regime 主导查询界（实际可用 per-regime θ 精化）；epoch 重置的 ACAL 会周期性丢弃全部历史——与"记忆保留"的直觉冲突，工程转写时应换成软衰减而非硬清空（理论只保证硬清空版本）；`D_t` 任意游走的对手性强于真实用户，速率参照偏悲观。

---

### 12. Learning with a Drifting Target Concept（ALT 2015）

**文件**：`papers/12-learning-drifting-target-concept-alt2015.pdf`（29 页；深读 §1–§6 全部定理、Theorem 1/3 证明主体与 §6 DriftingActive）

1. **基本信息**：Steve Hanneke, Varun Kanade, Liu Yang（字母序，杨柳第三）；ALT 2015。贡献声明："refining the best previous results for polynomial-time algorithms for linear separators under a uniform distribution ... efficient algorithm that achieves a bound on the error rate that is Õ(√d√Δ), where d is the VC dimension and at each time t the target function is allowed to change by at most some distance Δt."
2. **问题设定与核心结果**：固定分布 P，i.i.d. 无标注流，目标序列 `h*_t ∈ C` 每步漂移 `P(x: h*_t(x) ≠ h*_{t+1}(x)) ≤ Δ_{t+1}`；每步先预测再看标签，关心时刻 T 的误差 `er_T(ĥ_T)` 与累计错误数。背景图景（§3）：常数 Δ 时，可容忍漂移 `Δ_ε = Θ(ε²/d)` 即误差 `Θ̃(√(dΔ))`（Long 1999 上界，非高效算法；Helmbold–Long 的 BASICn 构造给出匹配下界——误差低于 `√(dΔ)/e²` 的 tracking 不存在，且 BASICn 可嵌入 halfspaces / axis-aligned rectangles）；已知多项式时间结果只有 `Õ(d√Δ)`（HL94 consistency-oracle 归约）与 `Õ((dΔ)^{1/4})`（CMEDV10 改造 Perceptron）。本文四个主结果：**Theorem 1（自适应窗口，任意 VC 类、任意未知 Δ 序列）**：算法不依赖 Δ，以概率 1−δ，`er_T(ĥ_T) ≤ O(min_{1≤m≤T−1}[ (1/m)Σ_{i=T−m}^{T−1}Σ_{j=i+1}^{T}Δ_j + (d·Log(m/d)+Log(1/δ))/m ])`——与"已知 Δ 的最优窗口 ERM"同阶（常数 Δ 时恢复 `Õ(√(dΔ))`，最优窗口 `m* ≍ √(d/Δ)`）。**Corollary 1 + Theorem 2（充要条件）**：`Σ_{t≤T}Δ_t = o(T)` ⇒ 期望累计错误 o(T)；反向对圆上齐次线性分类器成立——`ΣΔ_t ≠ o(T)` 时任意算法错误线性（构造：`φ_t = φ_{t−1} + min{Δ_t,½}π·B_t`，B_t 为 ±1 公平硬币，每步不可预测方向的旋转恰好造成 `min{Δ_t,½}` 错误率）。**Theorem 3（主技术贡献）**：齐次线性分类器 + 球面均匀分布 + 常数 Δ 时，poly(d,1/Δ,log(1/δ)) 时间算法达 `er_t(ĥ_t) = O(√(Δd·log(1/δ)))`，期望累计错误 `O(√(Δd)·log(1/(Δd))·T)`，主动学习版查询数 `O(√(Δd)·log^{3/2}(1/(Δd))·T)`——把高效算法从 `(dΔ)^{1/4}` 拉到与非高效最优同阶的 `√(dΔ)`。**Theorem 4（§6，一般类的主动学习）**：DriftingActive（A² 变体）达期望错误 `Õ(√(dΔ))·T`、期望查询 `Õ(θ_C(√(dΔ))·√(dΔ))·T`（θ_C 为最坏情形 disagreement coefficient）。**Jumps remark**：允许 K 次跳变（Δ_t=1）时错误界变 `Õ(√(dΔ)T + √(d/Δ)·K)`——每次跳变的恢复代价有界 `√(d/Δ)`。
3. **核心机制**：Theorem 1 的自适应窗口统计量是全篇最可移植的构件：`m̂_T = max{m : min_{h∈C} max_{m'≤m} [Σ_{t=T−m'}^{T−1}1[h(X_t)≠Y_t]] / [d·Log(m'/d)+Log(1/δ)] < K}`（K=145c²）——**向过去扩窗，直到"窗口内最优拟合的错误数超出纯估计噪声的量级"为止**，然后在选中的窗口上跑 ERM；证明用 Bernstein + "h\*_T 在最优窗口内错误数 ≈ 累计漂移"的双向夹逼。Theorem 3 的机制：以 `M = Θ(√(d/Δ)·log(1/δ))` 为批长（恰为漂移积累到噪声地板的时间尺度），每批 ModPerceptron 热启动（常数误差）后跑 ABL margin-based 定位——第 k 轮只在带 `|w_{k−1}·x| ≤ b_{k−1} ∝ 2^{−k}/√d` 内采样、在球 `‖v−w_{k−1}‖ ≤ r_k` 内极小化 hinge 损失 `ℓ_{τ_k}`——把批内累计漂移 O(ΔM) 当作 agnostic 噪声水平处理。DriftingActive 同样以 `M = ⌈c₁√(d/Δ)⌉₂` 为 epoch，倍增块 + 容差阈值 `T̂_k = log₂(1/√(dΔ)) + 2^{k+2}·2eΔ` 收缩版本空间。
4. **思想结合点**：这是本簇对 Appendable"多快必须刷新"最直接的一篇，三条定量规则全部可落到具体机制：(i) **有效记忆窗口公式 `m* ≍ √(d/Δ)`**——"新鲜度折扣该多狠"的第一性答案：窗口内累计漂移 `mΔ` 与估计误差 `d/m` 平衡处；超过 `√(d/Δ)` 拍之前的单用户证据对当前状态估计是**负资产**。对应机制：`personal-conditioning.md` §2.1 的 `freshness` 门（`freshness=0 ⇒ is_injectable=False`）与 `confidence × freshness` 幅度折扣、`continuum-memory.md` 的 decay 与 `promotion_threshold`（owner 可回滚自适应参数）——现行都是手调常数，本篇给出它们应当追踪的量纲：`√(容量/漂移率)`。(ii) **误差地板 `Ω(√(dΔ))`**——目标侧持续漂移下，任何有界状态系统对该用户的跟踪误差有不可消除下限；这直接约束 Gate 11 per-user continuity 与七日 N+1 判据的**预期效应量**：personalization 相对 stateless 的可分辨优势被 `√(dΔ)` 地板压缩，用户漂移越快、可检出增益越小——A1 重开 prereg（主线方案 §2.1）冻结最小效应时应引用此结构，避免把"用户在漂移"误判为"记忆无效"。(iii) **`ΣΔ_t = o(T)` 充要条件 + 跳变代价 `√(d/Δ)`**——恒速漂移的用户上"平均误差趋零的关系连续性"数学上不存在（措辞边界）；而 regime 切换（R14 的离散身份跳变）代价有界，每次约 `√(d/Δ)` 拍的重学预算——`continuum-memory.md` 的 slow→fast reset（`reset_nested_context()`，Gate 6 判定生产回滚路径用 copy-init）正是工程化的"跳变恢复"，本篇给它的收益上限定标：好的跳变处理最多省下 `√(d/Δ)` 拍的重学。
5. **算法结合点**：最可转写的是 **Theorem 1 的自适应窗口统计量 → consolidation/保鲜调度**。对象：memory owner 内部的 retrieval 窗口与 session-post consolidation 的聚合深度（`multi-timescale-learning.md` 的 session-post slow loop）、State-KV/v2 readout 参考统计的重估节律（主线方案 §0 不变量 7 的"参考统计冻结 train-split 拟合"多久该换血）。做法：维护"向过去扩窗的最优拟合失配计数"（失配 = prediction mismatch，PE 家族信号，R-PE 合法；**不是** judge/continuity 分数），窗口内失配超出 `Θ(d·log m)` 噪声量级即截窗——不需要知道 Δ，漂移率未知的问题（我们的实际情形）被该统计量原生解决。这是纪律 2 下唯一同时满足"信号合法 + 无需漂移率先验 + 任意 VC 类适用"的漂移自适应构件。**不可直接转写**：Theorem 3 的高效算法绑定球面均匀 + 齐次线性分类器几何（我们的 readout 空间不满足），只取其"批长 = `√(d/Δ)`"的节律结论；DriftingActive 的 epoch 硬重置同 #11 的顾虑，转写应为软衰减。假设匹配度核对：P 固定（与 #11 互补的缺口——真实流两者皆漂，联合情形无现成定理，见第 7 点）；标签 realizable（无独立噪声——我们的观测噪声要求把 Theorem 1 的常数放大，方向不变）；Δ_t 对手性有界（成立）。
6. **成熟度与档位**：**A**——`√(d/Δ)` 窗口、`√(dΔ)` 地板、`ΣΔ=o(T)` 充要条件、自适应窗口统计量四件套是本簇对"记忆保鲜"最高密度的可转化产出。
7. **风险与不适配**：Theorem 1 是逐时刻误差界，累计错误版本经 `δ_T = 1/m̃_T` 转换后只有期望界；高效结果几何特化严重；P 固定与 D_t 漂移不可同时处理（CMEDV10 允许双漂但界更弱 `(dΔ)^{1/4}`，联合最优 open）；把窗口统计量接到 CMS 时要防它成为第二 owner——必须实现为 memory owner 内部的调度参数（类比 `promotion_threshold`），经快照发布 telemetry，而非外部模块直改记忆内容。

---

### 06. Statistical Learning under Nonstationary Mixing Processes（AISTATS 2019）

**文件**：`papers/06-nonstationary-mixing-processes-aistats2019.pdf`（11 页；深读全文含 Theorem 1 完整证明与 §3 product process）

1. **基本信息**：Steve Hanneke, Liu Yang（字母序，杨柳第二）；AISTATS 2019。贡献声明："learning algorithm for nonstationary mixing processes, specifically β-mixing processes, and proved that for bounded VC subgraph classes, the cumulative excess risk grows sublinearly in the number of predictions."
2. **问题设定与核心结果**：Vapnik 的 general learning setting——F 为 `Z→[0,1]` 的一致有界 VC subgraph 类（伪维 d），流式：每步用历史产出 `f̂_t`、在新点 `Z_t` 上结算 `f̂_t(Z_t)`，指标是累计超额风险 `Σ_{t≤T} E[f̂_t(Z_t)] − Σ_t inf_f E[f(Z_t)]`（逐时刻最优为基准）。两条假设**联合放松 i.i.d.**：(2) 漂移和次线性 `Σ_{t≤T} Δ_t = O(T^α)`，α∈[0,1)，其中 `Δ_t ≥ ρ(P_t,P_{t−1})`，discrepancy `ρ(P,Q) = sup_{f∈F}|E_P f − E_Q f| ≤ TV`（Mohri–Muñoz Medina 的类相关漂移度量，比 TV 更贴学习目标）；(3) β-mixing `β_k = O(k^{−r})`，r>0（不要求分布收敛，只要求平均变化减速 + 时间相关衰减）。**Theorem 1（主结果）**：取 `m_t = ⌈(t−1)^{(1−α)(3+2r)/(3+3r)}⌉`、`k_t = ⌈(t−1)^{(1−α)/(1+r)}⌉`，跳采样 ERM `f̂_t = argmin_{f∈F} Σ_{s=1}^{⌊m_t/k_t⌋} f(Z_{t−s·k_t})` 达 `累计超额风险 = O(T^{(3+(2+α)r)/(3+3r)})`——对一切 α<1、r>0 次线性；r→∞（快混合）时指数 →(2+α)/3，r→0 时 →1。方法**只依赖单参数 α**（与 r），不需要逐步 Δ_t 序列——这是相对 Barve–Long / Mohri 系（需要 Δ 序列或其常数上界，后者排除次线性）的关键放松。**Theorem 2（product process、已知 Δ）**：独立非同分布时窗口 ERM 达 `Σ_t min_m (Σ_{q=t−m}^{t−1}Δ_{q+1} + √(d/m))`（有限样本、非渐近）。**Theorem 3（常数漂移 γ）**：固定窗 `m̄ = ⌈d^{1/3}γ^{−2/3}⌉` 达 `≲ (dγ)^{1/3}·T`——一般损失下的 Barve–Long 推广（注意与 #12 分类 realizable 情形 `√(dγ)` 的量纲差：一般损失是立方根）。§4 open problems：平稳情形已知最好 `O(T^{(3+r)/(3+2r)})` 用本文技术可复现（m_t=t−1），但一般 α=0 非平稳情形是否可达同率未知；非平稳 mixing 的 minimax 下界完全空白；general-loss 下对未知 Δ_t 的自适应亦 open（对照 #12 在分类下已解决）。
3. **核心机制**：三步解耦。(i) `f̂_t` 只依赖 `Z_{≤t−k_t}`，β-mixing 定义直接给 `‖P_{(f̂_t,Z_t)} − P_{f̂_t}×P_{Z_t}‖ ≤ β_{k_t}`——评估点与模型解耦的代价是 β_k；(ii) 跳采样块技术（Volkonskii–Rozanov/Eberlein/Yu）：间隔 k_t 的 n 个样本与独立版本的 TV 距离 ≤ `(n−1)β_{k_t}`，于是独立情形的 Rademacher 界 `c√(d/⌊m_t/k_t⌋)` 可用——**有效样本量是 `m/k` 而非 m**；(iii) 漂移项 `(1/⌊m/k⌋)ΣΣΔ` 经"每个 Δ_q 至多被 m_{2q} 个时刻的窗口覆盖"的重数技巧收成 `O(m_{2T}·ΣΔ_q) = O(T^{(1−α)(3+2r)/(3+3r)+α})`；三项同阶即最优调度，解出 m_t、k_t 的幂次。
4. **思想结合点**：本篇回答 Appendable 前提性的合法性问题——**对话流不是 i.i.d.，有界容量的持续学习还能不能有保证**：能，条件是时间相关多项式衰减（session 内强相关、跨 session 衰减的对话流符合此形状）+ 平均漂移减速。两条设计对应：(i) **cadence gating 的统计正当性**。`continuum-memory.md` 的 CMS 分层 cadence（`cadence_interval / observations_since_update`、session_medium/background_slow "不再每 turn 同频更新"）与 `multi-timescale-learning.md` 的 session-post slow loop，从本篇视角不是算力妥协而是统计必要：跳采样 k_t 就是"隔 k 拍才吸收一个学习样本"，相关样本的有效量按 `m/k` 折算，每 turn 同频更新对相关流没有额外统计收益、反而放大漂移项——**"慢层学得慢"有了收敛率层面的辩护**。session 长度 ≈ 自然的 k（跨 session 相关性弱于 session 内），background-slow 的更大间隔对应更大的 k_t 档位。(ii) **证据侧的镜像**。主线方案 §2.1/§2.2 的 dyad/scenario-clustered CI 是评估侧对相关性的折算，本篇是训练侧的同款折算；C3（主线方案 §4.3）的 episode 终局稀疏信用天然是大 k 间隔的学习信号——S3-E"稀而准"的信用性质从 mixing 角度获得第二重解释：稀疏化同时买到了去相关。
5. **算法结合点**：可转写对象 = **consolidation/学习调度的节律参数**。具体两条：(a) `multi-timescale-learning.md` 的 `JointLoopSchedule`（`ssl_interval / rl_interval / rl_batch_accumulation`）现为手调常数，本篇给出其应当追踪的量纲——间隔 `k ∼ horizon^{(1−α)/(1+r)}`、窗口 `m ∼ horizon^{(1−α)(3+2r)/(3+3r)}`：**随运行时长增长而加大间隔与窗口**（幂律调度），而非固定周期；(b) Theorem 3 的 `m̄ = d^{1/3}γ^{−2/3}` 是一般损失（如 N+1 表示 PE 的 MSE 型信用）下的窗口公式，与 #12 分类情形的 `√(d/Δ)` 并列成对——转写任何"用 PE 信用做在线学习"的窗口设计时，按信号类型选公式。信号来源核对：调度参数的输入只有时长与（估计的）α、r，不触 evaluation；α、r 的估计可从 owner 发布的快照指纹序列的自相关/漂移序列做只读诊断（类比主线方案 §0 不变量 7 的预检脚本形态，先诊断后冻结）。假设匹配度核对：**β-mixing 多项衰减**——对话流未验证，可测（block 自相关衰减拟合）；**ΣΔ = O(T^α) 且 α 已知**——α 未知是最大缺口，本篇自己承认 general-loss 下对 Δ 自适应 open；保守做法是取悲观 α 档位（牺牲率、保次线性）并把 α 的选择写进 prereg；**F 低伪维**——控制器层参数族满足，又一次指向低维架构。
6. **成熟度与档位**：**A**——它是三层对应里"非 i.i.d. 合法性"这一层的唯一承载，且 cadence gating 的统计辩护 + 幂律调度量纲可直接写进 06 综合文档的设计原则节。
7. **风险与不适配**：无 minimax 下界（调度公式可能远非最优，只作量纲参照）；渐近界无有限样本常数；α、r 双参数未知且误设时保证失效（α 设小了漂移项爆掉——保守方向是往大设）；discrepancy ρ 相对 F 定义，换 F（换控制器族）须重估；一般损失立方根率慢于分类平方根率，对 PE 型连续信用的窗口预期要按更悲观的 `(dγ)^{1/3}` 档位设。

---

## 簇级小结

### 一、对 Appendable 轴的净贡献清单（按可转化性排序）

1. **漂移自适应窗口统计量 → 记忆保鲜与 consolidation 调度**（#12 Theorem 1 + 常数 Δ 特例 `m* ≍ √(d/Δ)`；#06 Theorem 3 的一般损失版 `m̄ = d^{1/3}γ^{−2/3}`）。唯一同时满足"信号合法（PE 家族失配计数）+ 无需漂移率先验 + 任意 VC 类适用"的即插构件：向过去扩窗直到最优拟合失配超出 `Θ(d·log m)`，即为当前该信任的历史深度。落点：memory owner 内部 retrieval 窗口 / decay / `promotion_threshold` 的调度参数、State-KV `freshness` 折扣的量纲（`√(容量/漂移率)`）、v2 readout 参考统计的重估节律。
2. **诚实边界三件套**（#12 Corollary 1 / Theorem 2 + jumps remark；#11 Theorem 4）。`ΣΔ_t = o(T)` 是次线性错误的充要条件——恒速漂移用户上"平均误差趋零的连续性"不存在；目标漂移下跟踪误差地板 `Ω(√(dΔ))`——约束 Gate 11 与七日 N+1 判据的预期效应量；regime 跳变恢复代价有界 `√(d/Δ)`——slow→fast reset 类机制的收益上限。全部进对外表述禁语清单（见三）。
3. **cadence gating 的统计正当性 + 幂律调度量纲**（#06 Theorem 1/2/3）。相关流上有效样本量 = `m/k`，间隔化更新是收敛率层面的必要而非算力妥协；`ssl_interval / rl_interval` 类参数应随运行时长按幂律增长（`k ∼ horizon^{(1−α)/(1+r)}`）。同时给 S3-E"稀而准"信用第二重解释：稀疏终局信用天然去相关。
4. **群体先验闭环的存在性与成本模型**（#08 Theorem 1/3/4；#10 Theorem 2）。每用户至多 d 个额外证据即可支撑群体先验估计；估出的先验把新用户的适应成本压到"先验已知"水平、把**自验证**成本压到 o(1/ε)（无先验时可 Ω(1/ε)）；先验估计未收敛时的正确行为是回退保守通用路径（= cold-start 契约的理论版）。d 的最小性（#10 parity 构造）给盲判归属类证据的 probe 套件规模一个结构下限。
5. **低维 readout 架构的理论辩护 + 群体包功效核算工具**（#15 Theorem 1/2/3；#21 Theorem 4 + Lemma 1）。先验估计速率随维度以 `T^{−Θ(α²/d²)}` 崩塌 ⇒ 群体级学习必须在 16/14 维 typed readout、rank-3 z_t 这类空间进行；`E‖μ̂_ε−μ*‖ ≤ 3ε + 2√(ln|M_ε|/n)` 与 `T ≥ (64/ε²)ln N(ε/4)` 是可直接写进 prereg 的样本量公式；每用户证据深度在群体速率的两个指数档位（`α²/(2(d+2α)(α+2(d+1)))` ↔ `α/(d+2α)`）之间移动——衰减策略是群体学习速率的参数。
6. **环境漂移 vs 目标漂移的保留/衰减二分**（#11 全篇 ↔ #12 全篇）。分布漂移 + 身份稳定 ⇒ 记忆全保留 + 补覆盖采样（CMS promotion 沉淀跨情境不变结构）；目标漂移 ⇒ 有界窗口 + 衰减。regime 空间有效维度 m 定持续学习速率 `T^{m/(m+1)}`——保持 regime 表示低维有界有速率收益。

### 二、"先验可估性"对 Personal State-KV 证据线的理论支撑与边界

**支撑**：Personal 轨已 pass 的盲判归属（四轴 charter §8；`personal-conditioning.md` §3.4 P2 held-out pairwise 0.875、wrong-user 控制、跨 seed / 跨裁判稳定）在本簇理论中的位置是：#10 的可辨识性定理保证"状态→行为分布"的映射在联合多探针统计量上可逆（单探针可以完全无信息——P2 的成对多 probe 设计因此是必要而非偶然）；#08/#15/#21 则说明这条已验证的"个体可识别"通道向上可以支撑"群体先验可估"——16 维有界坐标空间的低维度恰好落在 #15 速率不至于崩塌的区间，`#21` 的线性分类器实例（方向参数化 + Hölder 密度）与 State-KV 的低秩方向几何同构。**边界**：(i) "π 可估"与"个体可识别"是两个命题，P2 pass 只证后者，前者需要独立的群体包证据；(ii) 全部先验定理假设任务间独立 + 共享 D + 目标任务内固定——用户间弱耦合、语境分布因人而异、用户随时间漂移三条都要在群体包 prereg 里作为威胁登记；(iii) 速率定理的 TV 度量强于下游决策质量，"先验估不准"不等于"个人化无收益"，反向亦然——群体包的判定门应定义在下游增益（如新用户 cold-start→warm-start 的 N+1 PE 改善）上，先验 TV 只做诊断。

### 三、漂移理论对"记忆保鲜 / consolidation 频率"的定量设计启示（含禁语清单）

设计启示浓缩为四条量纲规则：

| 量 | 公式 | 来源 | 落点 |
|---|---|---|---|
| 有效记忆窗口（分类型信号） | `m* ≍ √(d/Δ)` | #12 Thm 1/3 | freshness 折扣、retrieval 窗口、批长节律 |
| 有效记忆窗口（一般损失/PE 型信号） | `m̄ ≍ d^{1/3}·γ^{−2/3}` | #06 Thm 3 | PE 信用在线学习的窗口 |
| 跟踪误差地板 | `Ω(√(dΔ))` | #12（HL94 下界） | Gate 11 / 七日 N+1 最小效应定标 |
| 学习间隔（相关流） | `k ∼ horizon^{(1−α)/(1+r)}`，有效样本 `m/k` | #06 Thm 1 | `ssl_interval/rl_interval`、session-post cadence |

对外表述禁语清单（负结果同权）：

- **不能讲"记忆系统能让平均误差随时间趋零"**——除非用户漂移满足 `ΣΔ_t = o(T)`（#12 Theorem 2 的充要性对线性类成立且可嵌入更大类）；恒速漂移用户上该主张数学上为假。
- **不能讲"个人化增益可以任意大"**——目标漂移下相对 stateless 的可分辨优势被 `√(dΔ)` 地板与 `O(d log T)` 的无记忆基线（#11 Theorem 1：固定分布 realizable 时 CAL 本来就只错 `O(d log T)`）双向压缩；A1 类 prereg 的最小效应须按此定标，防止把"用户在漂移/任务太易"误判为"记忆无效"。
- **不能讲"每 turn 更新一定比隔拍更新学得多"**——相关流的有效样本量是 `m/k`（#06），高频更新对统计效率可以是零贡献。
- **不能讲"旧数据多多益善"**——超过 `√(d/Δ)`（或 `d^{1/3}γ^{−2/3}`）窗口的单用户历史对当前状态估计是负资产（#12/#06 窗口公式的另一面）；这是 decay 存在的第一性理由，不是容量妥协。
- **不能把本簇任何速率当保证引用**——#11 噪声情形与 #15 有界样本情形上下界均有 gap；#06 无任何下界；全部常数未量化；i.i.d./独立性/固定 P/固定 D 各有违反。引用一律"量级/结构参照"。

### 四、供 06 综合文档使用的转化候选表

| 论文 | 轴 | 对象 | 一句话方案 |
|---|---|---|---|
| #12 | Appendable | memory owner 的窗口/decay/`promotion_threshold` 调度 | 实现 Theorem 1 的扩窗统计量（PE 失配计数 vs `Θ(d·log m)` 噪声线）作 owner 内部可回滚调度参数，漂移率无需先验 |
| #12 | Appendable | Gate 11 / 七日 N+1 prereg 最小效应定标 | 用 `√(dΔ)` 地板 +（估计的）用户漂移率反推可检出增益区间，写进 prereg 效应量论证 |
| #06 | Appendable | `JointLoopSchedule` 与 session-post cadence | 间隔/窗口按 `k∼horizon^{(1−α)/(1+r)}`、`m∼horizon^{(1−α)(3+2r)/(3+3r)}` 幂律增长；α、r 由快照指纹序列只读诊断后随 prereg 冻结 |
| #06 | Learnable（跨轴） | C3 信用面的调度解释 | episode 终局稀疏信用 = 大 k 去相关采样，作为 S3-E 性质在对话域成立的 mixing 论证写入 C3 叙事 |
| #08 | Appendable | 群体先验包（远期）：新用户 warm-start | 以 16 维 readout 上的参数族为 π 家族，跨用户去标识快照估计，产物经 `ModificationGate` 作 cold-start→warm-start 初始化；判定门定义在新用户 N+1 PE 改善上 |
| #10 | Appendable | P2/P3 与 Gate 11 行为探针套件规模 | probe 数下限 = 用户状态族有效维度量级；readout 扩维时探针套件同步扩容，防"观察窗 < d"型假阴性 |
| #15 | Appendable | 群体先验包 prereg 的 power 分析 | 按 Theorem 1/2 上下界给用户数 T 的可行区间；先验密度光滑度在跨用户快照上做只读预检（对齐主线方案 §0 不变量 7 的预检纪律） |
| #21 | Appendable | 同上（有限样本工具） | 用 `3ε+2√(ln|M_ε|/n)` 与 `T ≥ (64/ε²)ln N(ε/4)` 出样本量表；MLE 替代 skeleton 解决可计算性 |
| #11 | Appendable | 漂移流上的昂贵 oracle 调度 + regime 表示设计 | matched settlement / 验证锚 unit 优先投向模型间分歧区（漂移下查询仍次线性）；regime 空间维度 m 保持低维有界（速率 `T^{m/(m+1)}` 随 m 退化） |
| #11 | Appendable | 保留/衰减二分规则 | 漂移诊断先分型：环境侧（身份稳定）→ 保留 + 补覆盖；目标侧 → 窗口 + 衰减；接入 reflection 的 consolidation 决策依据 |

### 五、跨簇引用

- **簇 1（#09/#07/#25/#24/#16 等）**：本簇的 #11 把簇 1 的"分歧驱动查询调度"结论延拓到漂移流；簇 1 的 i.i.d. 总注（其小结 §二末条）所指的"非平稳修正归簇 3"即本文 #06/#11/#12 三篇——两簇合读才构成完整的查询/学习预算理论。#08 Theorem 4 的自验证 o(1/ε) 与簇 1 的 #24 是同一结果的两个版本（#24 为独立成文的会议版），"先验补验证"的方法论在两簇小结中口径一致。
- **簇 2（#04 online selective sampling）**：#11 §6 的 querying-before-predicting 与 #04 的 mistakes-queries 权衡曲线同族；gate 的在线择时学习若要引用"先查后答"结构，以 #04 为准、#11 为漂移扩展。
- **博士论文**：第 11 章 §11.4.2（Random Drifts）含 #12 会议版没有的随机漂移分析（漂移方向随机而非对手性时界可改善），若"用户漂移是随机游走而非对手"的建模在群体包中被采纳，应回读该节。
