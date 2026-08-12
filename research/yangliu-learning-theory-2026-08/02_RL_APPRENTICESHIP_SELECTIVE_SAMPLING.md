# 02 · RL / 学徒学习 / 择时查询（簇 2，4 篇）

> 总索引：[00_PAPER_INDEX.md](00_PAPER_INDEX.md)。跨簇：簇 1 [01_ACTIVE_LEARNING_LABEL_COMPLEXITY.md](01_ACTIVE_LEARNING_LABEL_COMPLEXITY.md)、簇 5 [05_EARLY_ML_AND_THEORY_MISC.md](05_EARLY_ML_AND_THEORY_MISC.md)。
> 阅读口径：4 篇均逐篇深读（摘要/引言/主定理/算法骨架/关键引理，证明按需跳读）。
> Volvence 侧对照文档：[appendable-readable-learnable-steerable.md](../../docs/appendable-readable-learnable-steerable.md)（四轴 charter，重点 §4.2/§4.4/§5）、[主线提升方案_2026-08.md](../../docs/moving%20forward/主线提升方案_2026-08.md)（工作流 B/C）、[steering-runtime.md](../../docs/specs/steering-runtime.md)（三件套契约与 C1 终局信用链）、[steering-human-anchor.md](../../docs/specs/steering-human-anchor.md)（C2 验证锚）、[apprenticeship-alignment.md](../../docs/specs/apprenticeship-alignment.md)（已实例化的 ALT 2025 概念映射）。

---

## 簇定位：这 4 篇与 Steerable gate 择时、C2 验证锚预算的总关系

本簇是杨柳论文集里与 volvence 主线**形式同构度最高**的一簇。Volvence Steerable 轴的核心问题是「**何时**出手干预」：gate 在每拍面对二动作 `{noop, steer}`，当前处于 SHADOW，靠 C1 终局信用（N+1 表示 PE 的 matched noop-vs-action 比较，`appendable-readable-learnable-steerable.md` §4.2）做 bounded policy-gradient 学择时；C2 把少量昂贵的人类专家标注定位为验证锚（`validation_anchor_only=true`，§4.4），预算极小（pilot 48 units / 40 person-hours）。这个「大量自主行动 + 极少量昂贵可信查询 + 可靠性要求」的结构，正是本簇 #03（reliable active apprenticeship learning）与 #04（online selective sampling 的 mistakes–queries 权衡）研究的对象；#01（bandit 可学习性不可判定）为「系统主张判词能说到哪一步」提供数理纪律；#18（椭球版本空间）是三十年前的方法论远祖——用二阶摘要维护「与历史一致的假设集」，是 #03/#04 里 disagreement-based 版本空间的工程雏形。

### 结构同构表（reliable active apprenticeship ↔ steering gate）

| 论文概念（#03 设定） | Volvence 对应物 | 对应精度 |
|---|---|---|
| MDP 状态 `s_t`（无 reward） | 残差 belief：`steering_condition_belief` 快照（belief_label / margin / fresh_* / staleness_proxy / base_action_entropy，`steering-runtime.md` §4） | 结构级成立：两边都是「无内生 reward、按状态决定是否借助外部信号」 |
| 可查询 oracle 返回最优动作 `π*(s_t)` | 双通道：C1 的 N+1 terminal PE（免费、客观、滞后一拍）与 C2 人类锚（昂贵、双盲、方向性） | 部分成立：oracle 返回**动作标签**，C1 只返回**标量优势**、C2 只返回**方向验证**（见失效处 4） |
| query 动作（付出查询预算） | C2 送标（一个 annotation unit 绑定一个 gate `decision_id`，`steering-human-anchor.md` §2）；广义上也含 C3 的昂贵 matched counterfactual（每 turn 只算一次，`steering-runtime.md` §5） | 成立：两边的查询都是显式预算物 |
| reliable 保证（不查询时动作必须 = `π*(s_t)`） | strict noop 默认（noop ⇒ 逐元为零 delta）+ 保守干预（norm cap、无 free bias，`steering-runtime.md` §5） | **弱化后成立**：volvence 承诺的是「不干预时零改动」（安全性），不是「不查询时动作最优」（最优性）；见失效处 3 |
| disagreement 区 `DIS(V)` 触发查询 | `apprenticeship_alignment` owner 的 `reliability == DEFERRING` + `should_request_feedback`（`apprenticeship-alignment.md` §7.1）；gate 观测里的 `belief_disagrees_fresh` 是同族信号 | 已实例化（学徒 turn）；gate 侧只是同族启发，无版本空间语义 |
| eluder dimension `e_π*`（择时假设类的查询复杂度刻度） | 择时策略类（gate policy / 未来 z_t 空间控制器）的复杂度度量的**候选形式语言**；目前 volvence 无任何显式复杂度账本 | 思想级：连续 belief 空间上 `DIS(V)` 不可计算，`e_π*` 不能直接算，但「查询数下界由假设类组合结构决定」这一定性结论可迁移 |
| Massart 噪声专家（`P(π*(s)|s) ≥ max_{a≠π*(s)} P(a|s) + Δ`） | C2 双标注者一致性门（exact agreement ≥ 0.75、Cohen's κ ≥ 0.60，`steering-human-anchor.md` §7）；`apprenticeship_alignment` 的矛盾需 margin/recurrence 确认（不变量 4） | 成立且已部分实例化：两边都拒绝把单次低置信对立当真信号 |

### 同构失效处（必须写明，否则转写是表面类比）

1. **状态空间性质**：论文的下界构造依赖「状态可以被环境重复送回」（eluder 序列 / 驻留态 `P(s_n|s_n,a)=1`）；volvence 的 belief 空间是连续残差读出（896–3584 维），对话非平稳且同一状态**不可重访**。`DIS(V)` 的精确计算在此不可行，只能用 margin/uncertainty 类 surrogate——这正是 gate 观测四元组的现状，但它失去了论文的正确性证明。
2. **oracle 时不变性**：论文假设 `P(·|s)` 条件独立于历史且时不变；人类专家判断依赖对话上下文与时间，且 C2 的标注对象（steered vs noop 双臂盲评）本身是滞后离线材料，不是在线 oracle。
3. **reliable 语义漂移**：论文的 reliable = 不查询时**动作正确**（`a_t = π*(s_t)`），是最优性承诺；volvence 的 strict noop = 不干预时**不改变基底行为**，是保守性承诺。gate 选 NOOP 从不声称 noop 是最优动作。把两者混同会夸大 Steerable 的保证等级。
4. **反馈强度**：论文查询返回最优动作本身（监督标签，可直接收缩 `V`）；C1 返回的是 `clip((noop_mse − action_mse)/max(·), −1, 1)` 的标量优势且滞后一拍（episode 终局才结算，`steering-runtime.md` §5 C1 链），C2 返回的是「专家方向与 C1 方向是否一致」的聚合验证，均不是动作标签。论文的 `V ← {π : π(s_t) = a_t}` 收缩算子在 volvence 两条信号面上都没有直接对应物。
5. **契约级禁令**：论文的核心环路是「查询 → 更新版本空间 → 学习」；C2 的 `learning_use_authorized=false` 把这条环路**在契约上切断**（`steering-human-anchor.md` §1）。因此本簇一切「oracle 反馈进学习环路」的转写在近期都不合法，只能落在预算分配 / 一致性门 / 审计调度这些验证侧对象上（详见分析纪律）。
6. **策略类有限性**：#03 的界含 `log|Π|`（论文脚注 4 说明该依赖对有限动作集不可避免）；gate policy 是连续参数逻辑斯蒂策略，形式上 `|Π| = ∞`，需换 sequential 复杂度或有限化（如策略网格 / 版本号离散化）才能引用界。

---

## 分析纪律声明（先于一切结合点）

- **(a) C2 验证锚定位不可越界**：人类专家查询理论（#03 的 oracle、#04 的 label query）在 volvence 的**近期合法落点**只有三个——标注预算分配（该送哪些 `decision_id`）、双标注者一致性门（κ/agreement 阈值的理论依据）、审计调度（何时触发下一批 pilot）。任何「专家标签更新 gate / 收缩策略集」的方案都触碰 `validation_anchor_only=true` + `learning_use_authorized=false`（`steering-human-anchor.md` §1、`appendable-readable-learnable-steerable.md` §4.4），只能写成**远期单独 prereg 的理论准备**（对应主线方案 §4.2 的升级条件：仅当 C1 免费信用被证明与人类锚 load-bearing 不一致时，经 credit owner 正式入口另立 prereg）。本文所有「算法结合点」均按此分档。
- **(b) 不可判定性 = 诚实边界素材**：#01 的 ZFC 不可判定结果不用于任何机制设计，只用于 thesis 判词纪律——「不存在同时完备、显式、简单的可学习性刻画」支持 `thesis prove.md` 式的「可以说 / 不能说」二分口径（`appendable-readable-learnable-steerable.md` §8）。
- **(c) evaluation/judge 禁作学习信号**：本簇所有查询/信用转写的信号源只允许 PE 通道（C1 N+1 terminal PE）与人工标注通道（C2，且仅验证）；七日 continuity readout 与 companion-bench judge 分数不进入任何环路（R12，主线方案 §4.0）。
- **(d) 禁止 token 空间 RL**：本簇全部转写对象都在控制器层——gate 二动作 policy、未来 z_t 空间择时策略、C2 送标 selector——不触碰 token 生成层（R3/R4）。

---

## 逐篇深析

### 03. Reliable Active Apprenticeship Learning（ALT 2025）

**文件**：`papers/03-reliable-active-apprenticeship-learning-alt2025.pdf`（PMLR vol 272:1–27）

1. **基本信息**：Steve Hanneke（Purdue）、**Liu Yang**（第二作者，中国联通数字科技数据智能事业部）、Gongju Wang、Yulun Song（均联通数科）。杨柳贡献声明（原文）："I invented this topic ... in the noise free case, I gave the main algorithms for both the non-episodic case and the episodic case ... introduced a new complexity measure called Eluder-star dimension ... under the Massart noise condition and under the Tsybakov noisy condition, I respectively proposed the algorithms that are robust to the noise and proved the upper and lower bounds for sample complexity ... defined and formulated the important problem of Agnostic Apprenticeship Active Learning and designed an algorithm in the agnostic setting and proved the optimal sample complexity bound."
   **版本核对（诚实登记）**：正式 PMLR 版只覆盖 non-episodic 单 episode 设定下的 realizable / Massart / mixed-margin 三档；**Eluder-star dimension、episodic 情形、agnostic 算法均不在正式版内**——agnostic 在 §7 被明确列为 open problem（"Perhaps the clearest question is the extension to the agnostic setting ... A primary challenge ... is even to formulate what kind of reliability requirement is possible"）。贡献声明指向的应是扩展版/后续工作，引用时必须区分。

2. **问题设定与核心结果**：MDP 式环境 `P = (P0, P(·|s,a), oracle P(·|s))`，**无 reward**；未知最优策略 `π* ∈ Π ⊆ A^S`。learner 每拍可查询 oracle 获得建议动作；**reliable** 定义为：凡不查询的拍，动作必须满足 `a_t = π*(s_t)`（噪声情形放宽为以概率 ≥ 1−δ 全轨迹成立）。目标是最小化查询数。核心复杂度度量是策略类的 **eluder dimension**：`e_π0 = max{n : ∃s_1..s_n, ∀i ≤ n, s_i ∈ DIS({π ∈ Π : ∀j<i, π(s_j) = π0(s_j)})}`。主结果：
   - **Theorem 3（realizable，紧刻画）**：算法 ACAL 的查询数 ≤ `min{e_π*, T}`，且任何 reliable learner 在某确定性环境下查询数 ≥ `min{e_π*, T}`——最优查询复杂度**恰为** `min{e_π*, T}`。
   - **Theorem 6 + 8（Massart，gap Δ）**：ReliableApprentice 在 `P(π*(s)|s) ≥ max_{a≠π*(s)} P(a|s) + Δ` 下 reliable，查询数 `O(e_π* · Δ⁻² · log(|Π|T/δ))`。
   - **Theorem 9（Massart 下界）**：`Ω((e_π* + Δ⁻² log(1/δ)) ∧ T)`——δ 失败概率不可去除，且 `Δ⁻²` 项不可避免。
   - **Theorem 11/12（mixed-margin，Tsybakov 推广）**：作者提出 mixed optimal trajectory 上的 margin 条件 `(1/t)Σ 1[gap(s_t′) ≤ τ] ≤ C·τ^{α/(1−α)} + (1/t)log(1/δ′)`；上界 `O(e_π* · T^{(2−2α)/(2−α)} · log(|Π|T/δ)^{α/(2−α)})`，下界 `Ω((e_π* + T^{(2−2α)/(2−α)}) ∧ T)`，两界有 gap，Conjecture 18 猜下界紧。

3. **核心机制**：disagreement-based 版本空间收缩。ACAL 维护 `V = {π : 与所有已查询回答一致}`，仅当 `s_t ∈ DIS(V)` 时查询，否则执行 `V` 全体一致的唯一动作——`π* ∈ V` 不变量直接给出 reliability，查询序列恰构成以 π* 为中心的 eluder 序列，故 ≤ `e_π*`。噪声情形（ReliableApprentice）不能硬淘汰：改为保留所有与经验最优 `π̂_t = argmax_{π∈V} Σ 1[π(s)=a]` 差距在 empirical-Bernstein 置信带内的策略（鞅浓缩，Bernstein 差分序列对 `(s_1..s_t, â_1..â_{t−1})`），Massart 条件把「经验差距」转化为「与 π* 的 Hamming 距离」：`V` 始终含于以 π* 为球心、半径 `k = O(Δ⁻² log(|Π|T/δ))` 的 Hamming 球。最后 **Lemma 13（组合引理）** 把带罚版本的 disagreement 序列压回真 eluder 序列：若 `∀i, s_i ∈ DIS({π : Σ_{t<i} 1[π(s_t)≠π0(s_t)] ≤ k})`，则 `n ≤ (k+1)·e_π0`。三步合成 `|Q| ≤ (k+1)e_π*`。下界用 Anthony–Bartlett 的 Bernoulli 偏置检验（Lemma 14）+ 两环境不可区分构造。

4. **思想结合点**：这是全簇与 volvence 咬合最深的一篇，且**已经被实例化一次**：`apprenticeship-alignment.md` §2 的概念映射表（专家查询→operator teach、版本空间→意图约束集、可靠性区/不一致区→reliable/deferring、eluder 信息量→`guidance_surprise`、Massart→矛盾需 margin/recurrence 确认）就是以本文为理论主干建的 ACTIVE owner。它对 Steerable 轴的未竟结合点在 gate：`appendable-readable-learnable-steerable.md` §5.1 的第三层「学会何时扳」目前是纯 REINFORCE 学习问题，没有任何「查询预算的下界意识」；本文 Theorem 9 给出的定性结论是——**任何想要可靠性保证的择时者，查询数有不可压缩下界 `Ω(e + Δ⁻²log(1/δ))`**。翻译到 C2（§4.4 / 主线方案 §4.2）：如果未来要求人类锚对「gate 方向正确」给出某置信等级的验证，所需 unit 数由「专家噪声 gap Δ」与「待区分策略结构」共同下界，不能指望 48 units 的 pilot 覆盖任意强的主张——这为 pilot 只回答 rubric 可扩量性、方向一致率只在 ≥24 个 resolved & PE 非中性 unit 上解释（`steering-human-anchor.md` §7）提供了理论正当性叙事。
   另一个已兑现的映射核对：spec 把「eluder 信息量」松弛为逐条指导的 `guidance_surprise`（覆盖度的补），这是**借名而非定理转写**——论文中 eluder dimension 是策略类的最坏情形组合量（查询数刻度），不是单条样本的信息量；spec 的用法是合理的工程启发，但引用时不应声称其继承了论文的最优性保证。同样，spec 的 `version_space_status == INCONSISTENT`（塌空判矛盾）在论文 realizable 语义下不会发生（`π* ∈ V` 是不变量），它实际对应的是论文 §7 留白的 agnostic 情形——即 spec 在正式理论**尚未覆盖**的区域做了自己的工程延拓，这一点与贡献声明的「agnostic 已解决」互为印证候选，但在扩展版公开前应按 open problem 对待。

5. **算法结合点**：
   - **可转写（近期，验证侧）**：C2 送标调度。论文的查询规则「`s_t ∈ DIS(V)` 才查询」转写为「优先送标 gate 决策中 *择时不确定性最高* 的 `decision_id`」——具体信号已有：`belief_margin` 小、`belief_disagrees_fresh` 为真、`steer_probability` 接近 0.5 的 unit。这不改变 C2 的任何契约（送哪些 unit 本就是 packet builder 的自由度），却能让 48-unit 预算集中在信息量最大的决策上。核对：信号来源全部是 owner 快照字段（合法）；风险是送标分布有偏后，方向一致率不再是无偏总体估计——需要在 pilot report 里如实记录选样规则（对应 `steering-human-anchor.md` §7 的「不在 pilot 中途调参」，选样规则必须在 packet 构建前冻结）。
   - **可转写（远期，单独 prereg 的理论准备）**：若 C1 与人类锚被证明 load-bearing 不一致、专家标注按主线方案 §4.2 升级条件经 credit owner 进入学习环路，ReliableApprentice 的「empirical-Bernstein 置信带 + Hamming 球收缩」是带噪专家标签下更新择时策略的正确参照系（专家双标注 κ 门 ≈ Massart gap 的经验代理）。
   - **不可直接转写**：ACAL/ReliableApprentice 的主环路本身。理由三条：(i) `DIS(V)` 在连续 belief 空间不可计算（失效处 1）；(ii) 查询返回的不是动作标签（失效处 4）；(iii) 学习用途被 C2 契约禁止（失效处 5）。

6. **成熟度与档位**：**A**——理论主干已被 `apprenticeship_alignment` owner 实例化（学徒域），gate 择时与 C2 预算侧还有两个未兑现的具体落点（送标调度、下界叙事）。

7. **风险与不适配**：正式版三档噪声模型都假设 π* 点态最优且 oracle 时不变，对话域两者皆不成立；`log|Π|` 依赖要求有限策略类；reliable 语义与 strict noop 语义的差异（失效处 3）若不写明，会把 Steerable 的安全承诺误售为最优性承诺；贡献声明与正式版的范围差（Eluder-star/episodic/agnostic）在对外引用时必须注明。

---

### 04. Toward a General Theory of Online Selective Sampling: Trading Off Mistakes and Queries（AISTATS 2021）

**文件**：`papers/04-online-selective-sampling-aistats2021.pdf`（PMLR vol 130）

1. **基本信息**：Steve Hanneke、**Liu Yang**（第二作者，字母序；署名 TTI-Chicago）。贡献声明（原文）："I provided this general theoretical framework of online selective sampling. I also established the trade-off between the number of mistakes and the number of queries and proved properties of the optimal trade-off curve."

2. **问题设定与核心结果**：随机在线设定——`X_t` iid 采自未知 `P`，realizable（`Y_t = f*(X_t)`，`f* ∈ H`，VC 维 `d < ∞`）。learner 每拍必须预测 `Ŷ_t`，并可选择查询真标签（`Q_t ∈ {0,1}`）。研究对象是权衡曲线 `(M_T, Q_T)`（期望错误数 vs 期望查询数）在三个分析层级下的形状：
   - **Distribution-free（Theorem 1/3/4）**：定义 **trivial modifications of CAL**（TM(CAL)）——用与内容无关的确定性索引集 `I ⊆ N` 筛掉部分拍再跑 CAL。Theorem 3：任给查询预算 `q_T ≤ T`，存在 `A ∈ TM(CAL)` 使 `Q_T ≤ q_T` 且 `M_T ≤ d log T + (dT/q_T)·1[q_T < s·ln(eT)]`（`s` = star number）。Theorem 4（minimax 下界）：任何算法 `M_T ≳ min{d,T} + (T/Q_T)·1[Q_T < s/16]`。结论：分布无关层面，**最优权衡曲线整条被 TM(CAL) 近似占满**——想省查询，唯一通用代价表就是 `M_T ≈ T/Q_T` 的反比曲线（在 star number 预算内），聪明算法无本质优势。
   - **Distribution-dependent（Proposition 5 + Theorems 6/7）**：构造 VC 维 = 1 的双树类与分布 `P`，使 PickyActive（只在 disagreement 区与低质量子域相交处查询）达到 `Q_T ≲ log²T` 且 `M_T ≲ (T log T)^{1/2}`，而任何 TM(CAL) 只要 `Q_T < T^{1/17}` 就必有 `M_T > T^{7/8}`——**分布相关层面，内容感知的查询策略可以指数级优于内容盲筛**。
   - **通用 P-dependent 算法（PickySplitting，Theorem 8）**：把 Dasgupta 的 splitting index `ρ(ε;τ)` 搬进在线流，批大小序列 `{T_i}` 控制权衡，批内用 secretary-problem 停止规则挑 `Split(E,x)` 最大的点查询。
   - **Target-dependent 开放问题**：是否每个 VC 类都存在 `M_T = O(log T)` 且 `Q_T = o(T)` 的算法？未解；Theorem 9 证明 k-区间并类上 UIntActive_k 同时达到 `M_T = O(log T)` 与 `Q_T = O(log T)`。

3. **核心机制**：上界方向，CAL 的关键性质是「不查询即无信息损失」（不查询的拍标签可被完美推断），配合 `E[P(DIS(V_{t−1}))] ≤ s/t` 的 disagreement 质量衰减界，把筛选索引集 `I = {1..i_T}` 的错误代价折算为 `d·T/q_T`。下界方向用 star set 构造：`s` 个点每个只被一个假设翻转，任何少于 `s/16` 次查询的策略在剩余点上必然以 `T/Q_T` 速率积累错误。分布相关分离的构造是递归双树（高质量区低概率、低质量区高概率），让 CAL 的「查询一切 disagreement」浪费在低信息点上，而 PickyActive 只查高信息子树。

4. **思想结合点**：这是「**验证/标注预算 ↔ 系统错误率**」的形式化理论，直接对应 volvence 两个缺口：(i) **C2 扩量决策**——`steering-human-anchor.md` §7 规定 pilot 48 units 通过一致性门后「是否扩到 120–240 units 由独立 power/budget prereg 决定」，该 prereg 目前没有任何形式框架；本文的权衡曲线（查询预算 `q_T` ↔ 错误上界 `d log T + dT/q_T`）就是这类 prereg 的正确形式语言：预算翻倍能买到多少验证分辨力，应写成显式曲线而不是拍脑袋数字。(ii) **Gate 4 后续**——`apprenticeship-alignment.md` §7.2 记录 Gate 4 主动学习省标签 `not-supported`（五臂 balanced accuracy 全 0.5）与 v3 retest 的 labels-saved = −1.0。本文的三层级结构精确解释了这个负结果的可能位置：**distribution-free 层面聪明选样本来就不该赢**（Theorem 1：TM(CAL) 即近优，内容盲随机筛不可被本质超越）；聪明策略的优势只存在于 distribution-dependent 层（Proposition 5），且依赖「低概率高信息区」这种特定分布结构。Gate 4 的 segment 表示若没有创造这种结构，学不出 label gain 是理论预期而非工程失误——这为「后续应先改变 temporal segment representation 而非降低标签门槛」（spec 原话）补上了理论脚注。

5. **算法结合点**：
   - **可转写（近期，验证侧）**：C2/审计的**预算-分辨力曲线 prereg 模板**。对象：扩量 prereg（48 → 120–240）与远期持续审计调度（每 N 个 SHADOW turn 抽 k 个 decision 送标）。转写：把 Theorem 3/4 的形状（错误 ≈ 常数 + 预算的反比项、超过某复杂度阈值后追加预算无收益）作为 power 分析的结构先验；「TM = 内容盲筛」恰是 Gate 4 已有的 `random-sampling baseline` 臂（`apprenticeship-alignment.md` §7.1 #87），任何学习型送标 selector 必须先赢过它才配上线——这与本文「TM(CAL) 是基线，赢它需要分布结构」完全一致。核对：全部落在验证侧与 evidence 臂设计，不触学习环路，合法。
   - **可转写（结构启发）**：PickySplitting 的批内 secretary 停止规则——在「必须实时决定是否送标、不能回看」的在线审计场景（未来 ACTIVE 后的抽检）里，「用批首 `1/e` 段定标、之后首超即取」是无需分布知识的可行调度骨架。
   - **不可直接转写**：全部定量界。理由：realizable + iid 假设双双不成立（对话流非 iid，人类锚有噪声）；`M_T` 在 volvence 无直接对应物（gate 的「错误」要经 C1 PE 结算才可见，且滞后一拍）；开放问题本身未解，说明该理论对我们最关心的 target-dependent 渐近区仍是空白。

6. **成熟度与档位**：**A**——不提供可搬算法，但提供 C2 扩量 prereg 与审计基线设计的**形式框架**，且是解释 Gate 4 负结果的最佳理论透镜。

7. **风险与不适配**：iid + realizable 是强假设；界中 `d`、`s` 对 volvence 的择时策略类无现成算法可算；论文自认是「first steps」，权衡曲线的一般理论（尤其 target-dependent）未完成，引用时不得当成熟理论使用。

---

### 01. Bandit Learnability can be Undecidable（COLT 2023）

**文件**：`papers/01-bandit-learnability-undecidable-colt2023.pdf`（PMLR vol 195:1–38）

1. **基本信息**：Steve Hanneke（Purdue）、**Liu Yang**（第二作者，字母序；署名 Santai Technology）。贡献声明（原文）："I established a fully-general theory of bandit learnability and prove that bandit learnability is undecidable within ZFC set theory. I also used teaching dimension (an active learning complexity measure) to characterize the optimal query complexity for bandit learning for binary reward."

2. **问题设定与核心结果**：无噪声结构化 bandit——arm 空间 `X`，真实 reward 函数 `f* ∈ F ⊆ [0,1]^X`，拉 arm `x` 即观测 `f*(x)`；`F` 可学习定义为存在算法与 `M: (0,1)→N`，对一切 `f* ∈ F` 在 `M(ε)` 次查询内返回 `f*(x̂) ≥ sup_x f*(x) − ε`。主结果：
   - **Theorem 1（不可判定性）**：取 `X = R`，存在显式构造的 `F` 使「`(X,F)` 是否可学习」**独立于 ZFC 公理体系**。Theorem 2 证明可学习 ⟺ no-regret 可学习，故 no-regret 可学习性同样不可判定（Corollary 1）。
   - **二值 reward 的完备刻画**：确定性 learner——**zero-teaching dimension** `τ⁰_F = min{t : ∃x_1..x_t, min_{f∈F\{0}} max_i f(x_i) = 1}`，Theorem 3：可学习 ⟺ `τ⁰_F < ∞` 且 `M(ε) = τ⁰_F − 1`。随机 learner——**maximin volume** `σ̃_F = sup_P inf_{f∈F\{0}} P(x: f(x)=1)`，Theorem 4：可学习 ⟺ `σ̃_F > 0` 且 `(1−ε)/σ̃_F − 1 ≤ M(ε) ≤ ⌈σ̃_F⁻¹ ln(1/ε)⌉ − 1`；精确刻画由 randomized zero-teaching dimension `τ̃⁰_F(ε)` 给出（Theorem 5：`M(ε) = τ̃⁰_F(ε) − 1`）。Example 1 给出确定性/随机的无穷分离（余有限支撑类：`τ⁰_F = ∞` 而 `τ̃⁰_F(ε) = 1`）。
   - **实值推广**：level-set teaching dimension `τ_F(ε)`（确定性上界 `M(ε) = O(τ_F(ε)/ε)`，Theorem 6）与 maximin level-set volume `σ̃_F(ε)`（随机上界 `M(ε) ≤ (2/ε)⌈σ̃_F(ε/4)⁻¹ ln(2/ε)⌉`，Theorem 7）；线性类上 `τ_F(ε) = ε^{1−d}` 不紧（自适应分支序列变体 `≤ d+1` 才紧），Hölder 类上 `⌈(L/ε)^{d/α}⌉` 与已知最优吻合。

3. **核心机制**：不可判定性经由与 Ben-David et al.（2019）的 **EMX 学习问题**（用 iid 样本找 `ĥ` 使 `E_P[ĥ] ≥ sup_h E_P[h] − ε`）建立等价：对 union-bounded 子族，为每个分布 `P` 构造 reward 函数族，使每个 `h ∈ H` 对应 arm `x_h` 且 `f*(x_h) ∝ E_P[h(X)]`；技术核心是（i）用附加 arm 集 `x_w` 的 reward 值经 learner 自身随机性模拟 `P` 的 iid 采样，（ii）反向将 bandit learner 压成 **weak monotone compression scheme**——这是 EMX 不可判定性的机关所在：monotone compression 的存在性等价于关于连续统基数的集合论命题（`2^{ℵ₀}` 与 `ℵ_ω` 的关系），在 ZFC 内不可判定。二值刻画部分则是初等的：`τ⁰_F` 显然刻画确定性查询数，随机情形用 minimax 论证连接 `σ̃_F` 与 `τ̃⁰_F`。§1.6 与 DEC（Foster et al.）比较：E2D 的上界对确定性/二值 bandit 恒为无穷（Hellinger 覆盖数发散），下界对 singleton 类给不出 `Θ(T)` regret——印证「上下界 gap 不可消除」正是 Theorem 1 预言的必然。

4. **思想结合点**：两个落点，均在纪律与形式语言层，不在机制层。
   - **判词纪律（主落点）**：论文的方法论结论——「任何完备的可学习性刻画必然复杂到对集合论公理敏感，文献应转向对**明确子族**给出完备刻画」——与 volvence 的诚实边界实践同构：`appendable-readable-learnable-steerable.md` §8 的「可以说 / 不能说」二分、主线方案 §0 的「封存判词不可改写、退出条件先行」都是在拒绝无限定的一般主张。#01 把这种纪律从工程审慎升格为数学必然（详见簇级小结第二段）。
   - **最小验证集合的形式语言**：teaching dimension 度量「把目标从假设类中教出来所需的最少样本点」；C2 的「至少 24 个 resolved 且 PE 非中性 unit 才允许解释方向一致率」（`steering-human-anchor.md` §7）本质上是在问同一个问题——区分「C1 方向可信 / 不可信」两个假设最少需要多少昂贵观测。teaching dimension 是把这类门槛从经验数字升级为「相对被验证假设类的组合量」的候选语言。
   - **随机化的价值**：Example 1 的确定性/随机无穷分离提示：验证锚的 unit 抽样若引入随机化（相对确定性 checklist），可能以小预算覆盖确定性方案无法覆盖的失败模式——这与 C2 packet 的 A/B 随机分配已有的盲化设计一致。

5. **算法结合点**：**不可直接转写，且不应转写**。理由：(i) 无噪声确定性 reward 的设定与 volvence 任何信号面（PE 有噪、人类锚有噪）都不匹配；(ii) `τ⁰_F`/`σ̃_F` 的计算需要显式函数类，volvence 的「假设类」（gate 策略族、C1 方向假设）无此表示；(iii) 不可判定性结果本身是元定理，无算法内容。唯一的操作性使用是**否定性的**：当未来有人提议「为 volvence 的在线持续主动学习给一个完备的可学习性判据」时，本文是引用即驳回的依据——判据只能对预注册的明确证据族（A2 的 501 dyad、C3 的判定门结构）成立，不能对「系统能否学习」这类全称命题成立。

6. **成熟度与档位**：**A**——不进机制，进 thesis 判词纪律与 C2 门槛的形式语言储备；作为「负结果同权」的典范登记。

7. **风险与不适配**：不可判定性依赖病态构造（union-bounded EMX 类 + 连续统基数），与工程可实现类无关，**禁止**引申为「volvence 的学习问题不可判定」之类的修辞——正确用法只限「一般刻画不存在 ⇒ 主张必须限定子族」这一步；teaching dimension 落点目前只是语言候选，没有任何 volvence 假设类被形式化到可计算它的程度。

---

### 18. Online Learning by Ellipsoid Method（ICML 2009）

**文件**：`papers/18-online-learning-ellipsoid-icml2009.pdf`

1. **基本信息**：**Liu Yang（一作**，CMU 机器学习系博士期）、Rong Jin（MSU，其硕士导师）、Jieping Ye（ASU）。贡献声明（原文）："online learning algorithm using the ellipsoid method approximating the classification hypotheses that are consistent with all the training examples by an ellipsoid ... efficient algorithms for updating both the centroid and the positive definite matrix ... evaluated with USPS and three UCI datasets."

2. **问题设定与核心结果**：在线二分类，假设存在 γ-margin 分类器 `u`（`‖u‖=1`，`y_i uᵀx_i ≥ γ`）。与只维护单点解的主流在线算法（Perceptron/PA/MIRA）不同，本文维护**与全部历史一致的假设集的椭球外逼近**：一致集 `A_t = {z : y_i x_iᵀz ≥ aγ, i ≤ t}` 包含以 `u` 为心、半径 `(1−a)γ` 的球（Lemma 1），用椭球 `E_t = {z : (z−w_t)ᵀP_t⁻¹(z−w_t) ≤ 1} ⊇ A_t` 表示。两个算法：
   - **CELLIP**（可分情形）：误分类时以半平面 `C_t = {z : y_t x_tᵀz ≥ aγ}` 切割，更新 `w_{t+1} = w_t + α_t P_t g_t`、`P_{t+1} = (1−α_t²)P_t − 2α_t(1−α_t)P_t g_t g_tᵀ P_t`（`α_t = (aγ − y_t w_tᵀx_t)/√(x_tᵀP_t x_t)`，`g_t = y_t x_t/√(x_tᵀP_t x_t)`），体积比 `vol(E_{t+1})/vol(E_t) = (1−α_t²)^{(d−1)/2}(1−α_t)`；错误界由体积下界 `vol(B)` 顶住（Theorem 3）。
   - **IELLIP**（不可分情形）：放弃可行性叙事，把 `(w_t, P_t)` 重释为训练流的一/二阶统计摘要——`P_t⁻¹` 是加权协方差 `θ₀P₁⁻¹ + Σθ_i g_i g_iᵀ`；更新 `P_{t+1} = (1−c_t)⁻¹(P_t − c_t P_t g_t g_tᵀ P_t)`，**记忆参数** `c_t = c·b^{t−1}` 使旧样本权重指数衰减。以 `q_t = (u−w_t)ᵀP_t⁻¹(u−w_t)` 为势函数（Lemma 2），得错误界 `M ≤ 1/γ² + (2/γ)·(1−b)/(1−b−c)·Σ_i l_i(u)`（`l_i(u) = max(0, γ − uᵀx_i)`，Theorem 4；`c=0` 退化为标准界）。多标签扩展沿 Crammer–Singer 框架；USPS + 三个 UCI 数据集上与 PA/MIRA 可比或更优。

3. **核心机制**：把凸优化椭球法的「梯度切割」换成「误分类样本的 margin 半平面切割」，并证明存在一族 `(ρ, μ)` 参数使新椭球覆盖交集（Theorem 1 的约束 `(1−α²)/μ² + ρ²/(1−α−ρ)² ≤ 1`）。方法论意义在于：**版本空间不必显式枚举，可用 `O(d²)` 的二阶几何体近似维护**，且「中心 = 当前决策、形状矩阵 = 剩余不确定性」——这是 #03/#04 里组合版本空间 `V` 的连续参数化前身，也是她后来转向「与 oracle 交互的数学理论」的方法起点（对照簇 5 文档的谱系段）。

4. **思想结合点**：两个思想级落点。(i) **不确定性的显式二阶表示**：volvence gate 的观测（`belief_margin`、`base_action_entropy`）是标量化不确定性；IELLIP 提示「决策参数 + 不确定性形状」可以合并为一个可增量更新的对象，其中 `P_t` 的主轴方向指出「哪个方向的观测最能收缩不确定性」——这正是 C2 送标调度想要的「哪个 decision 最有信息量」的几何语言（与 #03 的 `DIS(V)` 判据同源，但连续可算）。(ii) **遗忘因子对非平稳的工程处理**：`c_t = c·b^{t−1}` 的指数记忆衰减是对「目标漂移」的最朴素响应，与簇 3 的 drifting target concept 理论（#12）互为工程/理论两面；volvence 的对应物是时间尺度分层本身（online-fast 控制器参数 vs 冻结基底，`AGENTS.md` §2），不需要引入该机制，但「二阶摘要 + 可调记忆」是未来 z_t 空间控制器状态表示的候选形态之一。

5. **算法结合点**：**不建议直接转写**。理由：(i) 线性可分/hinge 结构假设与残差 belief 空间的读出几何无对应；(ii) `O(d²)` 的 `P_t` 在 896–3584 维残差上是 0.8M–13M 参数的满矩阵，与 gate 当前 4 观测的轻量策略相比不成比例；(iii) gate 学习已有 C1 契约约束（bounded policy-gradient、一批至多 +1 policy_version，`steering-runtime.md` §5），引入椭球更新意味着换学习器，须走新 prereg，而没有证据显示现有 REINFORCE 是瓶颈。保留的唯一操作性想法：若未来 C2 送标 selector 需要「版本空间几何」而非启发式分数，低秩椭球（对 belief 的低维投影维护 `P_t`）是比组合 `DIS(V)` 可行得多的实现路径。

6. **成熟度与档位**：**B**——思想级参照（版本空间的连续二阶摘要、记忆衰减），无近期落点；作为 #03/#04 方法论谱系的起点登记。

7. **风险与不适配**：2009 年结果，错误界弱于后续理论；椭球外逼近在高维的体积效率差（`d² − 1` 因子每步侵蚀）；论文无查询/选择性采样成分（每拍都看标签），与本簇主题的连接是谱系性的而非问题级的。

---

## 簇级小结

### 一、对 Steerable（gate 择时）与 C2（验证锚预算）的净贡献清单（按可转化性排序）

| # | 贡献 | 来源 | 近期落点（合法性已核对） |
|---|---|---|---|
| 1 | **不确定性驱动的 C2 送标调度**：优先送标 `belief_margin` 小 / `belief_disagrees_fresh` / `steer_probability≈0.5` 的 `decision_id`，选样规则随 packet 构建前冻结 | #03 ACAL 的 `DIS(V)` 查询判据（Theorem 3 的最优性直觉）+ #18 的连续几何实现路径 | C2 packet builder 的选样策略（验证侧，不触学习环路；须在 pilot report 记录选样偏差） |
| 2 | **C2 扩量 prereg 的预算-分辨力曲线框架**：48 → 120–240 units 的 power/budget prereg 用「错误 ≈ 常数 + 预算反比项、超过复杂度阈值后追加无收益」的曲线结构写成显式函数 | #04 Theorems 3/4（`M_T ≤ d log T + dT/q_T`、star number 饱和阈值） | `steering-human-anchor.md` §7 预留的独立 power/budget prereg |
| 3 | **查询下界叙事**：任何 reliable 择时者的查询数下界 `Ω(e + Δ⁻²log(1/δ))`——为「pilot 只回答 rubric 可扩量、≥24 resolved unit 才解释方向一致率」提供理论正当性；也预告「零人类预算的可靠性主张」不存在 | #03 Theorem 9 | C2 文档叙事与 review 抗辩（非代码） |
| 4 | **审计基线纪律**：学习型送标 selector 必须先赢内容盲随机筛（TM = random baseline），且理论预期它只在存在特定分布结构时才赢——解释 Gate 4 `not-supported` 为理论一致而非工程失误 | #04 Theorem 1（TM(CAL) 分布无关近优）+ Proposition 5（分布相关分离条件） | Gate 4 后续与任何未来 selector evidence 的臂设计（`active-learning-off` random 臂先例已有） |
| 5 | **带噪专家更新的参照系**（远期，单独 prereg 的理论准备）：若 C2 升级为信用源，empirical-Bernstein 置信带 + Hamming 球收缩（Lemma 13）是带噪标签下更新择时策略类的正确形式 | #03 ReliableApprentice + Lemma 13 | 仅理论储备；触发条件 = 主线方案 §4.2 升级条件 + R-C2 分歧复审 |
| 6 | **最小验证集合的形式语言**：teaching dimension 作为「证伪一个方向假设最少需要多少昂贵 unit」的组合语言候选 | #01 Theorems 3/5（`τ⁰_F`、`τ̃⁰_F(ε)`） | C2 门槛数字（24/48）的远期形式化（非行动项） |
| 7 | **版本空间二阶摘要**：低秩椭球作连续 `DIS(V)` 替代物的实现候选 | #18 IELLIP | 无近期落点，登记备查 |

### 二、Bandit 不可判定性对系统主张判词纪律的启示

#01 的元结论——bandit 可学习性的完备刻画在 ZFC 内不存在，文献必须转向「对明确子族给出完备刻画」——是 volvence 判词纪律的数学同构物。Volvence 的主张体系（`appendable-readable-learnable-steerable.md` §8「可以说/不能说」、`thesis prove.md` §6 清单、主线方案 P4 的四项独立 prereg）本质上就是在执行同一策略：**放弃「系统能在线持续主动学习」的全称判定，只对预注册的、判定门结构冻结的证据族（A2 的 501 dyad quality+scaling 门、C3 的择时 admission、B3 的有序前缀晋升）逐一给出可判词的子命题**。#01 说明这不是工程保守，而是一般主张在数学上就没有简单完备的判据可依——任何声称「一个判据判定系统可学习性」的提案，其判据要么不完备（有 gap，如 DEC 的上下界），要么复杂到不可操作。因此对外表述的正确形态永远是「在子族 S 上、判据 G 下、判词 V」三元组，这也是为什么 A1 的 `passed=false` 必须限定为「v1 readout 下无净增益」而不是「七日窗口无净增益」（主线方案 §9 2026-08-12 记录）——判词范围限定不是措辞谨慎，是不可判定性时代做理论主张的唯一诚实方式。

### 三、供 06 综合文档使用的转化候选表

| 论文 | 轴 | 对象（owner/slot/prereg） | 一句话方案 |
|---|---|---|---|
| #03 Reliable Active Apprenticeship（ALT 2025） | Steerable / Learnable（验证侧） | C2 packet builder 选样策略；`steering_gate_decision` 快照字段为信号源 | 用 gate 择时不确定性（margin 小 / lagged-fresh 分歧 / steer_probability≈0.5）做 C2 送标优先级，选样规则 packet 前冻结并随 report 披露 |
| #03 同上（Theorem 9 下界） | Steerable（叙事） | C2 pilot 文档 / thesis 抗辩 | 引用查询数下界说明可靠性主张必然消耗人类预算，48-unit pilot 的解释范围限定有理论依据 |
| #04 Online Selective Sampling（AISTATS 2021） | Learnable（验证预算） | `steering-human-anchor.md` §7 预留的扩量 power/budget prereg | 以 mistakes–queries 权衡曲线（常数项 + 预算反比项 + 饱和阈值）为扩量 prereg 的形式模板 |
| #04 同上（TM 基线 + 分布相关分离） | Learnable（evidence 臂设计） | Gate 4 后续 / 任何送标 selector evidence | 学习型 selector 的对照臂必须含内容盲随机筛；只有证明数据存在「低概率高信息」结构才预期 selector 赢 |
| #01 Bandit Undecidable（COLT 2023） | 全轴（判词纪律） | `thesis prove.md` 口径 / P4 对账 | 完备可学习性判据不存在 ⇒ 主张只能是「子族×判据×判词」三元组；驳回一切全称学习能力判据提案 |
| #01 同上（teaching dimension） | Learnable（远期语言） | C2 一致性门数字的形式化 | 用 teaching dimension 语言把「≥24 resolved units」类门槛表述为相对方向假设类的组合量（登记，非行动项） |
| #18 Ellipsoid Online Learning（ICML 2009） | Readable/Steerable（远期表示） | 未来送标 selector / z_t 控制器状态 | 低秩椭球（中心=决策、形状=不确定性）作连续版本空间摘要的实现候选（登记，非行动项） |

### 诚实边界（本簇分析自身的）

- #03 贡献声明中的 Eluder-star dimension、episodic、agnostic 三项**未见于正式 PMLR 版**（agnostic 明确为 open problem）；本文分析只基于正式版内容，扩展版公开前不得引用这三项。
- 本簇全部「结合点」中，唯一已运行的实例化是 `apprenticeship_alignment` owner（学徒域，ACTIVE）；gate 择时与 C2 侧的候选均为**未执行的分析建议**，不构成任何 prereg、WiringLevel 变更或 formal 判词。
- Gate 4 的 `not-supported` / labels-saved = −1.0 负结果按原判词保留；#04 提供的是解释框架，不是翻案依据（同一 locked 分区不得调参重跑）。
