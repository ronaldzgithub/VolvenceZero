# C4 · 测试与审计理论簇（3 篇）

> 隶属：`research/yangliu-learning-theory-2026-08/`（总索引见 [00_PAPER_INDEX.md](00_PAPER_INDEX.md)）。
> 本簇 3 篇：#14 Active Property Testing（FOCS 2012）、#23 Testing Piecewise Functions（TCS 2018）、#05 Small Connectivity via Local Cut Algorithms（SODA 2020）。
> 阅读方式：每篇精读摘要/引言/模型定义/主定理与核心引理（前 10–15 页），按需跳读证明骨架，未逐页读完整证明。
> 本文件为只读分析文档，不改变任何 spec/契约/代码；文中一切"转写候选"均须按 AGENTS.md §8 收敛包纪律另立包并先冻结 prereg 才能落地。

---

## 0. 簇定位：testing ≪ learning 层级与 Readable 轴的审计面

Property testing 研究的问题是：**用远少于学习所需的查询数，判定一个函数 f 是否具有性质 P（f ∈ P），或距离该性质 ε-远（dist_D(f, P) ≥ ε）**。它与 PAC 学习的关系是"验证 vs 学出"：学习要输出一个 ε-好的假设，测试只要回答一个 yes/no——而后者在很多类上可以便宜一个多项式量级。本簇三篇合起来向 volvence 引入三件工具：

1. **testing ≪ learning 的查询复杂度分离**（#14/#23 的正结果）：区间并 active 测试 O(1/ε⁴) 查询 vs 学习 Ω(d)；Gaussian 下线性分离器测试 O(√n·log n) vs 学习 Ω(n)；k-分段函数 active 测试查询数与段数 k 无关。**验证可以在数量级上便宜于学习，且便宜多少是定理，不是口号。**
2. **testing dimension 作为"验证预算"的刻画工具**（#14）：对给定 (P, D)，存在组合量 d_passive / d_coarse / d_active，在常数因子内刻画测试所需的固有标签数——如同 VC dimension 之于学习样本复杂度。验证预算不是拍脑袋的，它有内在标度，且上下界都可证。
3. **下界意识**（#14 的负结果 + #05 的单侧错误语义）：dictator functions 的 active 测试需要 Ω(log n) 查询，与学习一样贵——**"验证便宜"不是普遍规律，是逐性质成立的事实**；同时 tester 的单侧保证方向（报告发现必真、报告未发现存疑）与 `docs/specs/evaluation.md` §风险类评估的"未触发 ≠ 安全证据"同构。

### 0.1 对应 volvence 的三个具体审计场景

Volvence 的 Readable 轴主张"内部状态可从残差与快照命名地读出"（`docs/appendable-readable-learnable-steerable.md` §3.1），配套证据纪律是：**每个 readout/仪器在用于 formal 判词前，先证明自己有判据分辨力**。这在结构上正是一个 property testing 问题——用有限只读证据判定"这台仪器具有'能分辨'这一性质"，而且必须比 formal run（"学习级"支出）便宜得多。三个具体场景：

| 场景 | volvence 锚点 | 被经济化的"查询"资源 | testing 理论的对应物 |
|---|---|---|---|
| (a) reader/sensor heldout 预检（读得对吗） | charter §3.2 / §5.1：S3-前置 reader heldout 1.0（PASS）后 `steering_condition_belief` 才 SHADOW 上线 | heldout 标注样本数 | learn-then-validate 分解（#23 §3 末）：validate 预算 O(1/ε)，与 reader 复杂度无关 |
| (b) 仪器判据分辨力预检 | `主线提升方案_2026-08.md` §0 不变量 7（共模能量占比、same-vs-diff Cohen's d、1-NN 检索）+ 不变量 8（传导预检）；A1 v1 readout d=0.315 失准 → v2 d=0.592（§9 2026-08-12 记录） | MPS/GPU formal 预算（预检本身零新增 GPU，用已收集 train-split） | testing dimension 的两假设族判别形式化：d_S(π, π′)——预检 = 判定"有信号世界"与"失准世界"在有限观测下是否可分 |
| (c) 人类验证锚预算量级 | 主线方案 §4.2 C2：pilot 双标注者 + inter-rater 一致性门 + 量级预算，validation-anchor-only | 专家标注对数 | closeness/agreement testing：验证"方向一致率 ≥ 1−ε"需 Θ(ε⁻¹~ε⁻²·log(1/δ)) 对，与被验 gate 的参数量无关 |

诚实注记：testing 理论直接刻画的只有第一类资源（标签查询数）；(b)(c) 中把"formal run 数""标注对数"读作查询是**结构类比**，其合法性来自两处形式同构——预检的三个统计量本质是有限样本上的分布可分性度量（对应 d_S），锚验证本质是一致率估计（对应 closeness testing 的平凡 Chernoff regime）——而非定理的直接适用。

### 0.2 分析纪律声明

1. **理论假设差距必须诚实标注**。本簇结果的假设与 volvence 现实之间有四条固定沟壑，每篇的"风险与不适配"逐条核对：(i) 查询模型——标准 membership query 允许对任意虚构点请标签，volvence 只有有限标注预算 + 分布内已收集样本（最多对应 active/passive 模型，绝不对应 membership 模型）；(ii) 函数类假设——定理逐类成立（区间并/LTF/分段函数），volvence 的 reader 是**单个冻结线性头**，"读得对"是对参照标签的一致性而非语法类成员资格；(iii) 无噪 oracle——理论假设标签精确，人类锚有噪（inter-rater 门存在的原因），噪声下需要 tolerant testing，其理论严格更难且与 active 模型不可比（#14 Thm A.7）；(iv) ε-gap 语义——dist ∈ (0, ε) 的对象无任何保证，预检门槛必须显式规定 gap 区处置。
2. **不许把"testing 便宜"夸大为"我们的验证已经便宜"**。本簇给的是设计原则（验证/学习预算分离、预算可刻画）与下界意识（有些性质验证不便宜、passive 有 √ 型下界），不是对 volvence 任何现有预检成本的背书。凡引用具体界，必须核对所在 regime（active vs passive）与假设匹配度。
3. **evaluation 只读不回灌（R12）**。本簇一切审计/预检视角都落在 evaluation-readout 侧或 prereg 前置侧：预检结论决定"许不许开 formal"，永不进入 PE/credit/ModificationGate 学习环路（`docs/specs/evaluation.md` 关键不变量；主线方案 §0 不变量 2）。

---

### 14. Active Property Testing（FOCS 2012）

**文件**：`papers/14-active-property-testing-focs2012.pdf`（arXiv:1111.0897v2）

1. **基本信息**：Maria-Florina Balcan（Georgia Tech）、Eric Blais（CMU）、Avrim Blum（CMU）、Liu Yang（CMU ML）。字母序，杨柳第四位。贡献声明（原文）："I was involved in designing testing algorithms for testing unions of d intervals and testing linear separators in R^n over the Gaussian distribution. As one of the major parts of this work, I developed a general notion of the testing dimension of a given property to characterize the intrinsic number of label requests needed for testing and used it to prove lower bounds for linear separators and dictator functions."——testing dimension 是她署名的核心贡献。

2. **问题设定与核心结果**。距离定义 `dist_D(f, P) = min_{g∈P} Pr_{x∼D}[f(x) ≠ g(x)]`。s-sample q-query ε-tester（Def 2.1）：从 D 抽 s 个 unlabeled 样本，对其中 q 个请求标签，f ∈ P 时以 ≥ 2/3 概率 Accept，dist_D(f, P) ≥ ε 时以 ≥ 2/3 概率 Reject。三个模型是同一定义的参数化：s 无限 = 标准 membership-query 测试 [Rubinfeld–Sudan]；q = s = passive 测试 [Goldreich–Goldwasser–Ron]；s = poly(n) = 本文提出的 **active testing**（对齐主动学习：只能对 unlabeled pool 中的点请标签，不能虚构查询点——医疗诊断例：可以给真实病人做检测，不能问"如果他不吸烟还会得病吗"）。主定理：
   - **区间并**（f: [0,1]→{0,1} 为 ≤ d 个区间的并）：active 测试 O(1/ε⁴) 查询、**与 d 无关**、对任意未知分布成立（Thm 3.1）；均匀分布下 unlabeled 仅 O(√d/ε⁵)。对照：学习需 Ω(d)（VC = 2d），passive 测试 Θ(√d)（上界为其新结果，下界 Kearns–Ron）。此前即使在 membership 模型也只有放松版（区分 d 区间并 vs 距 d/ε 区间并 ε-远）。
   - **线性分离器（LTF）over Gaussian**：active 与 passive 均 O(√n·log n) 标注样本（Thm 4.1），学习需 Ω(n)；下界 Ω̃(n^{1/3})（active）与 Ω̃(√n)（passive），近匹配。
   - **testing dimension**（§6）：d_passive、d_coarse、d_active 在常数因子内刻画对应模型的固有标签数（Thm 6.2/6.4/6.6）。
   - **dictator functions**（f(x) = x_i）：active 测试需 Ω(log n) 查询（Thm 6.7），与学习同阶——测试 dictator 与学习 dictator 一样难；由归约传导出 decision trees、低 Fourier degree、juntas、DNF 的 active 测试下界。
   - **模型分离**（App A）：Q ≤ Q^a ≤ Q^p 且两个不等号都可严格（dictator 分离 standard/active；区间并分离 active/passive）；tolerant 与 active 不可比（Thm A.7）；active 与 distribution-free 不可比（Thm A.10）。

3. **核心机制**。(i) 区间并测试器 = noise sensitivity 特征化：`NS_δ(f) = Pr_{x, y∼δx}[f(x) ≠ f(y)]`。正向 Prop 3.3：d 区间并 ⟹ NS_δ(f) ≤ dδ（每条边界被 (x,y) 跨越的概率 ≤ δ/2）；逆向 Lemma 3.4（证明核心）：取 δ = ε²/32d，NS_δ(f) ≤ dδ(1+ε/4) ⟹ f 距区间并 ε-近——经"局部自校正"构造：f 与宽 2δ 均匀核卷积得 f_δ，τ-阈值化去局部噪声，未定义区向左延拓，证明所得 g 是 ≤ d(1+ε/4) 区间并且 ε/2-近。测试器抽 O(1/ε⁴) 个 δ-近邻对估计 NS，按 dδ(1+ε/8) 阈值判决。(ii) LTF 测试器：自相关统计量 `ρ(f) = E_{x,y}[f(x)f(y)⟨x,y⟩]`，Matulef et al. 特征化 |ρ(g) − W(E g)| ≤ 4ε³ ⟹ g ε-近 LTF。难点：独立对估计 ρ 需 Θ(n) 样本（⟨x,y⟩ 典型量级 √n）；解法：q 个样本内全部 C(q,2) 个**非独立对**复用 + 截断 1[|⟨x_i,x_j⟩| ≤ τ]（τ = √(4n·log(4n/ε³))）+ Arcones 的 U-statistic Bernstein 型集中不等式，核心事实是对多数 y，(E_x[f(x)x·y])² 很小（Fourier 分解证明）⟹ O(√n·log n)。(iii) testing dimension：固定 (P, D, ε)，Π₀ = 支撑在 P 内的函数分布族，Π_ε = 质量 1−o(1) 落在 ε-远函数上的分布族；对样本 S 定义诱导标签分布距离 `d_S(π, π′) = (1/2)Σ_y |π_S(y) − π′_S(y)|`（TV 距离）。d_passive = 最大 q 使 `sup_{π∈Π₀} sup_{π′∈Π_ε} Pr_{S∼D^q}(d_S(π,π′) > 1/4) ≤ 1/4`；active 版把"从 poly 大小 pool U 中自适应挑 q 个点判别"形式化为深度 q 决策树类 DT_q 对混合分布 Fair(π, π′, U) 的最优判别误差。**下界证明模板 = 显式构造两个世界**：dictator 下界取 π = 均匀 dictator、π′ = 均匀随机函数，把 S 看作 q×n 布尔矩阵，π_S(y) = (等于 y 的列数)/n，Chernoff + 双重 union bound 证明 q < (1/2)log n 时所有列型近均匀 ⟹ 不可分；LTF 下界同模板，π = w∼N(0,I) 的 sgn(w·x)、π′ = 随机标签，可分性归结为随机矩阵算子范数集中 [Vershynin]。

4. **思想结合点**。(i) **场景 (b) 的形式化母体**：主线方案 §0 不变量 7 的分辨力预检，做的正是 testing-dimension 式判别——固定 readout 观测通道后，检验"仪器有分辨力的世界（same-dyad 对更近）"与"失准世界（same/diff 不可分）"在 583 个已收集样本诱导的观测分布上是否可分。Cohen's d、1-NN 检索准确率是 d_S(π, π′) 的实用代理；v1 → v2 修复（`docs/specs/prediction-error-loop.md` 变更日志 2026-08-12：55.4% 共模能量 + L20 独占 73% 把 d 压到 0.315，逐层归一 + 减参考均值 + 去 top-1 PC 后 0.592）在这套语言里就是：**换观测通道，把同一性质在给定证据量下的"有效 testing dimension"降下来**——通道选错时，不是效应不存在，是判别所需样本量被人为抬高（A1 v1 null 无法区分"无效应"与"仪器失准"，正是 §9 2026-08-12 判词限定范围的原话）。(ii) **下界意识**：dictator 反例说明"预检应当轻量"是需要逐性质论证的主张。不变量 7 要求"门槛随 prereg 冻结，不过门不许开 formal"——本文补充的姿态是：冻结门槛时应一并论证**该样本量下两族本来可分**（预检的预检），否则预检自身可能在"testing dimension > 可用样本量"的 regime 里空转。(iii) **双世界构造 = 负对照设计模板**：与 evaluation spec §风险类评估"未触发 ≠ 安全证据"同源——判别失败永远有两种解释（无效应 / 仪器盲），构造 π′（已知无信号世界，如 shuffled/swapped 对照）是把两种解释拆开的唯一手段；A1 的 swapped-user-state 臂（实测 5.93e-05）与不变量 8 的传导预检正是这个模板的实例。

5. **算法结合点**。可转写对象一：**预检协议的样本量论证框架**（详见簇级小结）——把不变量 7 预检形式化为两假设族判别，prereg 附一段"n = 583（或所在域实际样本量）下，若真实 d ≥ 门槛值，则判别功效 ≥ 1−δ"的功效论证；这是把 testing dimension 的"固有查询数"思想降级为标准功效分析，假设匹配度高（只需 i.i.d. 与效应量定义，不需函数类假设）。可转写对象二：**线性 reader 漂移哨兵的量级参考**——LTF 测试 O(√n·log n) ≪ 学习 Ω(n) 提示：监测"残差 → 标签关系是否仍线性可读"原则上不需要重新采集重学 reader 那么多标注（n = 896 或 2688 维时，√n 量级 ≈ 30–52 vs n 量级 ≈ 10³）；**假设核对**：该定理严格依赖坐标独立标准 Gaussian——volvence 残差经 L2 归一落在球面、v2 又减均值去主成分，边缘分布非 Gaussian，论文明言一般分布下 LTF 测试开放。故只可作数量级直觉与"先测后学"的流程灵感，不可把 √n·log n 写进任何 prereg。不可直接转写：membership 模型的 O(1) 测试器（任意点查询在 volvence 中无 oracle——无法向现实请求虚构用户状态的真值标签）。

6. **成熟度与档位**：**A**——testing dimension 与 testing ≪ learning 分离是本簇支柱；双世界负对照模板与"预检的预检"直接塑造审计预算观。

7. **风险与不适配**：(i) active tester 可**自适应**挑点请标签，volvence 预检是冻结 train-split 上的只读被动统计——所在 regime 是 passive 测试，引用预算直觉时必须用 passive 侧（√ 型），不得偷换成 active 侧（O(1) 型）。(ii) 无噪 oracle 假设：人类锚有标注噪声，tolerant testing（接受近性质对象）才是现实语义，而 tolerant 与 active 不可比（Thm A.7），这段理论缺口真实存在、不可含糊。(iii) 性质 = 语法函数类 vs volvence 的"读得对" = 对参照标签的一致性——后者最接近的形式是对已知 g 的 closeness testing，落在平凡 Chernoff regime，本文的高级结果（U-statistic、自校正）多数用不上。(iv) ε-gap：dist ∈ (0, ε) 无保证；预检门槛的 gap 区处置当前是"不过门即不开 formal"（安全侧），转写任何 tester 时须保持这一保守方向。

---

### 23. Testing Piecewise Functions（TCS 2018）

**文件**：`papers/23-testing-piecewise-functions-tcs2018.pdf`

1. **基本信息**：Steve Hanneke、Liu Yang。字母序，杨柳第二位（二人论文，Hanneke 是其 16 篇合作的最长期合作者）。贡献声明（原文）："established the query complexity of property testing for general piecewise functions on the real line, under a zero-measure crossings condition. I also proved that, in the active testing setting, the query complexity of testing general piecewise functions is independent of the number of pieces."

2. **问题设定与核心结果**。X = R，值域 Y 任意（配 σ-代数）。k-分段函数类：`F_k(H) = {f(·; {h_i}, {t_i}) : h_1..h_k ∈ H, t_1 ≤ … ≤ t_{k−1}}`，f 在 (t_{i−1}, t_i] 上取 h_i。基类 H 满足**零测度交叉条件**：`∀ h ≠ h′ ∈ H, λ({x : h(x) = h′(x)}) = 0`（式 (1)）——多项式（p+1 点定一多项式）、平移 sine、平移正态 pdf 均满足。复杂度参数 = H 的 **graph dimension** d：集合族 {{(x, h(x)) : x ∈ X} : h ∈ H} 的 VC dimension（分段常值 d = 1；p 次多项式 d = p+1）。距离 ρ(f, g) = P(f ≠ g)，P = Uniform(0,1)（绝对连续分布经经验 CDF 重标定归约，graph dimension 与零测度交叉均保持）。主定理：
   - **Theorem 1（active，一般 H）**：存在 s-sample q-query ε-tester，`s = O((dk/ε⁶)·ln(1/ε))`，`q = O((d/ε⁸)·ln(1/ε))`——**查询数与段数 k 无关**（k 只进 unlabeled 预算）。
   - **Theorem 2（piecewise constant）**：`s = O(√k/ε⁵)`，active q = O(1/ε⁴)，passive q = s；且 ε ∈ (0, 1/8) 时任何 passive tester 需 `s = Ω(√k)`——passive 的最优 k-依赖恰为 √k（下界经"区间并 ⊂ 分段常值"的归约继承 Kearns–Ron/#14）。
   - k < 80/ε 时改走 **learn-then-validate**（任意分布 P 成立）：`s = q = O((dk/ε)·ln(2ek)·ln(1/ε)) + O(1/ε)`；其中 F_k(H) 的 graph dimension ≤ 4dk·log₂(2ek)。
   - **Open problem**：p 次多项式测试的最优 p-依赖（active/passive/membership 三模型全开放）；平凡上界 p + 1 + (1/ε)ln 3。作者特别指出"能拟合任意 p+1 个点"不构成 Ω(p) 下界的理由——分段常值同样能拟合任意 k 个点，却可 √k / O(1) 测试。

3. **核心机制**。(i) **广义 noise sensitivity**（对一般值域的关键推广）：`NS_δ(f, x; H) = inf_{h ∈ H_{(x, f(x))}} P(h(x′) ≠ f(x′) | x)`，其中 H_{(x,y)} = {h : h(x) = y}，x′ ∼ Uniform(x−δ, x+δ)；H_{(x,f(x))} 空时取 1；NS_δ(f; H) = E_x[·]。直觉：局部邻域内"有 H 中函数能续上 f 在 x 的行为"的失配率——二值情形退化为经典定义。(ii) 正向 Lemma 1：f ∈ F_k(H) ⟹ NS_δ(f; H) ≤ (k−1)δ/2（失配只能来自 k−1 个切点被 (x, x′) 跨越，每点概率 ≤ δ/4 × 双向）。逆向 Lemma 2（证明核心）：δ = ε²/32k，NS_δ(f; H) ≤ (k−1)(δ/2)(1+ε/4) ⟹ ρ(f, F_k(H)) < ε。原 #14 证明的"平滑 + 取整"对一般 Y 无意义，重构为**投票**：对每个 h 定义 `f_δ^h(x) = (1/2δ)∫ 1[f(t) = h(t)]dt`（示性函数与均匀核卷积），零测度交叉 ⟹ Σ_h f_δ^h ≤ 1 ⟹ 至多一个 h 得票 > 1/2；得票 ≥ 1−τ 处定义 g*(x) = h_x(x)，未定义区向左延拓；1/(2δ)-Lipschitz 平滑性 + NS 预算给出转变点数 m ≤ (k−1)(1+ε/2)，再把 F_{m+1}(H) 投影回 F_k(H)（删质量最小的 m+1−k 段，代价 < ε/2）。(iii) 估计器与一致收敛：每个种子 x_i 取 ℓ = O((d/ε⁴)ln(1/ε)) 个 δ-邻居，`N̂S_δ(f, x_i; H) = min_{h ∈ H_{(x_i, f(x_i))}} (1/ℓ)Σ_j 1[h(x′_ij) ≠ f(x′_ij)]`——inf 要对整个 H 同时估准，靠 VC relative deviation bound（`A_{ℓ,m} = 4(d·ln(2eℓ/d) + ln(96m))/ℓ`）；m = O(1/ε⁴) 个种子平均后按 (k−1)(δ/2)(1+ε/8) 阈值判决（Lemma 3 双向蕴含）。(iv) piecewise constant 特化：常值段使 inf 消失（NS_δ(f) = P(f(x) ≠ f(x′))），单对统计量 `N̂S′_δ(f) = ((1−2δ)/m′)Σ_r 1[f(z_r) ≠ f(y_r)]` 即可，m′ = O(1/ε⁴) 对；**生日悖论配对**：把样本按块分组，每块 n = 1 + ⌈2√⌈1/δ⌉⌉ 个 i.i.d. 点以 ≥ 1/2 概率自然产出一对 δ-近邻 ⟹ unlabeled 总量 O(√k/ε⁵)（√k 来自 √(1/δ) = √(32k)/ε）。

4. **思想结合点**。(i) **场景 (a) 的理论母体**：learn-then-validate 分解就是 S3-前置 reader 协议的形态（charter §3.2：冻结线性 ridge reader 在 train 上 fit，heldout 上验证 1.0 后才授权 SHADOW owner 化）——理论增量在于预算的**不对称性有定理**：fit 预算随复杂度走（O(dk/ε·polylog)），validate 预算只随 (ε, δ) 走（O(1/ε·log(1/δ))），与被验对象的维度/段数无关。这给"验证锚可以远小于训练集"（C2 的 validation-anchor-only 定位，主线方案 §4.2）以定理级依据。(ii) **场景 (b)/(c) 的量级骨架**："active 查询数与 k 无关"翻译为审计语言：**审计成本不必随被审对象的结构复杂度增长**，只要有廉价 unlabeled 池制造统计量需要的近邻对——而不变量 7 预检的 same-vs-diff dyad Cohen's d 正是一个对统计（pair statistic），其对构造成本可用生日配对逻辑核算。(iii) **R3/R4 时间抽象的审计接口**：β_t 切换 ⟹ z_t/belief 轨迹分段常值，"轨迹是否 ≤ k 段分段稳定"恰是 F_k(常值) 成员性测试的离散时间版——本文给出该性质的正式统计量（近邻对不一致率）与预算（与 k 无关的 O(1/ε⁴) 对）。

5. **算法结合点**。可转写对象一：**验证锚/heldout 预算的量级论证**（场景 (a)(c)）。两条可直接写进 prereg 的规则：(1) heldout 全对（1.0）的正确解读——n 个样本零错误在 95% 置信下只授权 err ≤ 3/n（rule of three，learn-then-validate 中 validate 步 Chernoff 界的特例）：S3-前置的 heldout 集大小 n 决定了"reader 读对"这句话的效应量分辨率，预注册最小可检 ε 应由 n 反推，而不是反过来；(2) C2 pilot 的方向一致性验证按 agreement-rate 检验设计——区分"一致率 ≥ 1−ε"vs"≤ 1−2ε"需 Θ(ε⁻¹·log(1/δ))（单侧、近 realizable）至 Θ(ε⁻²·log(1/δ))（双侧估计）标注对，与 gate/reader 参数量无关；inter-rater 门先估标注噪声 η，有效效应按 (1−2η) 收缩后再定对数。假设匹配度：这两条只用 i.i.d. + Chernoff，是本簇中匹配度最高的转写。可转写对象二：**段结构审计原型**（SHADOW 诊断脚本级）——对 `steering_condition_belief` 的 belief_label 轨迹定义近邻 turn 对（|t − t′| ≤ δ_turns）不一致率 N̂S′，对照 (k−1)δ/2 型预算判"分段稳定 vs ε-远"；unlabeled = 已记录轨迹（免费），"查询" = 需 fresh-read/人工核验的拍数，O(1/ε⁴) 对且与段数无关。假设核对：时间轴离散均匀（近似 Uniform 成立）、常值段标签互异（零测度交叉对离散标签退化为真）、但 ε-远的拒绝语义须先冻结离散化距离定义——可行、需 prereg 显式化；且它只审计"结构是分段稳定的"，不审计"分段位置语义正确"。不可直接转写：Theorem 1 的一般 H 版本（对 H 全类做经验 inf + 一致收敛——volvence reader 是单个冻结函数不是函数族）；ε⁻⁸ 字面常数在真实预算下不可搬，只搬分解结构。

6. **成熟度与档位**：**A**——learn-then-validate 与 k-无关查询是三个审计场景中 (a)(c) 的直接理论母体；生日配对给对统计的 unlabeled 预算公式；段结构审计是本簇唯一"可落到具体 slot 的统计量"级候选。

7. **风险与不适配**：(i) 一维实线设定，高维推广是论文明示的 future work——volvence 的时间轴恰好一维（贴合），但任何"残差空间中的分段性质"无理论覆盖，不得外推。(ii) 无噪 oracle 同 #14；主定理还要求 k ≥ 80/ε（小 k 走 learn-then-validate 分支）——转写时先核对段数与 ε 的适用域。(iii) ρ 是 0/1 不一致测度：belief margin/staleness 等连续量的"段"边界模糊，只有经 owner 发布的离散标签（belief_label）才能套用——这恰好强制走快照契约而非 consumer 自行阈值化（R8 顺向约束）。(iv) 该审计对"k 是多少"不敏感是优点也是盲区：它验证不了"恰好 k 段"，只验证"≤ k 段可解释"；若审计目标是段数本身，需另立统计量。

---

### 05. Computing and Testing Small Connectivity in Near-Linear Time and Queries via Fast Local Cut Algorithms（SODA 2020）

**文件**：`papers/05-small-connectivity-local-cut-soda2020.pdf`（arXiv:1910.14344，两篇 arXiv 稿合并）

1. **基本信息**：Sebastian Forster（Salzburg）、Danupon Nanongkai（KTH）、Thatchaphol Saranurak（TTIC）、Liu Yang（Independent Researcher）、Sorrachai Yingchareonthawornchai（Aalto）。字母序，杨柳第四位。贡献声明（原文）："randomized algorithm spending O(k²ν) time and O(kν) queries for a variant of local cut-detection problem in a directed graph ... property testing algorithms for k-edge and k-vertex connectivity with query complexities near-linear in k. This resolved two open problems (open for 20 years), one by Goldreich and Ron [STOC '97] and one by Orenstein and Ron [TCS '11]." 注：这是图算法/性质测试社区的合作，与她的学习理论主线（oracle 交互的数学理论）关联较弱，属其博士后期的延伸方向。

2. **问题设定与核心结果**。**LocalEC**：有向图邻接表查询访问，给定种子 x、体积 ν、割参数 k、松弛 γ ≤ k < ν < m(γ+1)/(130k)，判定是否存在 L ∋ x 使 |E(L, V−L)| < k 且 vol_out(L) ≤ ν；输出割集 S（保证 |E(S, V−S)| < k+γ、vol_out(S) ≤ 130νk/(γ+1)）或 ⊥。Theorem 3.1：随机算法 **O(νk²/(γ+1)) 时间、O(νk/(γ+1)) 边查询**，单侧错误——存在合法 L 时以 ≥ 3/4 概率输出割（≤ 1/4 概率误报 ⊥）；不存在时输出的任何割仍按构造正确。γ = 0 即贡献声明的 O(k²ν)/O(kν)；γ = ⌊εk⌋ 给 (1+ε)-近似版 O(νk/ε) 时间。改进 Chechik et al. 的 O(k^{O(k)}ν)。三个应用：(i) 无向图点连通度 Õ(m + nκ³)（κ = polylog(n) 时近线性，Aho–Hopcroft–Ullman 1974 教科书开放问题的里程碑进展）；(ii) **性质测试**（Thm 1.3/1.4）：k-边/点连通性，unbounded-degree incident-list 模型 Õ(k/ε²) 查询、bounded-degree Õ(k/ε)，简单图 k-edge 另有 Õ(min{k/ε², 1/ε³})——此前最好界 Õ((ck/εd)^{k+1}) 对 k **指数**，本文降到近线性，解决 Goldreich–Ron（STOC '97，局部 Karger 算法在 testing 相关参数域的改进）与 Orenstein–Ron（TCS '11，k 的多项式化）两个约 20 年开放问题；(iii) 极大 k-边连通子图 Õ(k^{3/2}m^{3/2})（对 k 从指数降到多项式）。

3. **核心机制**。(i) LocalEC 算法极简：重复 k+γ 轮"带随机急停的 DFS"——从 x 生长 DFS 树，每标记一条新边以概率 (γ+1)/(8ν) 急停，把树上 x → 当前点的路径**整体反向**；若某轮 DFS 未被急停即自然耗尽，返回可达集 V(T)；累计标记边数超 128νk/(γ+1) 即 ⊥。(ii) 正确性支点 Lemma 3.5（Chechik et al. 的观察）：反向一条 x→y 路径，y ∉ S 时 |E(S, V−S)| 与 vol_out(S) **恰好各减 1**，y ∈ S 时不变——于是若小割 S′ 存在，至多 k−1 次"逃出"就耗尽割边，此后 DFS 必被困在 S′ 内并返回它；两种 ⊥ 事件（预算超限 / 全部 k+γ 轮完成）分别用负二项分布期望 + Markov 与"逃出次数 ≤ γ"的期望论证压到各 ≤ 1/8。(iii) 测试算法的结构（方法论价值所在）：**ε-far 的组合特征化**（Thm 6.5，引 [OR11] Cor 8）——`G ε-远于 k-边连通 ⟺ 存在互斥子集族 {X_1..X_t} 使 Σ_i (k − d_out(X_i)) > εm（或 in 版本）`：全局性质违反**等价于**一笔可加的局部亏损预算，且见证互斥。算法据此按亏损档 2^i（i ≤ log k）与体积档分桶，证明存在某档的"小体积见证"足够多（Lemma 6.6：|C_{i*,small}| ≥ εnd/(4k(⌊log k⌋+1))），于是均匀抽 Θ̃(k/(εd)) 个种子以常数概率命中某个见证，再对每档参数 (ν_i, γ_i = min{2^i −1, ⌊k/2⌋}) 跑 LocalEC 局部证实。加上"抽 Θ(1/ε) 个点查度数 < k"的平凡割检查兜底。

4. **思想结合点**（诚实弱连接）。与 Readable 审计面的**直接**结合弱：volvence 没有需要 k-连通性判定的运行时结构，本篇价值是方法论三件套：(i) **全局性质违反 ⟺ 有界体积局部见证的可加分解**——审计一个大工件不必全量遍历，先证明"若整体坏，则存在足够多的小局部坏证据"，再抽样 + 局部证实；(ii) **未知见证参数用对数个倍增档覆盖**（亏损 2^i × 体积 2^j 双重分桶）——不知道违规"多大"时，枚举 log 个尺度而不是精确搜索；(iii) **预算封顶 + 显式 ⊥ + 单侧错误**：查询预算超限即返回 ⊥ 而非静默继续（fail-loudly 的算法版）；且错误方向固定——**输出割必真实（可当场验证），输出 ⊥ 存疑**——与 `docs/specs/evaluation.md` §风险类评估的"未触发 ≠ 安全证据"完全同向：审计工具应优先"报告违规必真、报告通过存疑"的单侧保证，把存疑方向留给重复采样压低。松弛参数 γ 的"精度换查询预算"显式旋钮（O(νk/(γ+1))），与主线方案 §2.2 A2 scaling 门三选一裁决（用打平门 + 斜率门替代不可测的渐近门）在"显式声明放松了什么来换可判性"上仅为思想同构。

5. **算法结合点**。**无直接可转写对象**，理由：(i) 查询模型是邻接表边查询 q(v, i)，volvence 审计对象（快照、artifact、内存卡片）的成本瓶颈不是均匀边访问；(ii) ε-far 的距离语义（编辑 εm 条边）没有自然对应物——快照 DAG 的"违规"是契约布尔而非编辑距离；(iii) 当前 slot 依赖图仅数十节点，全遍历平凡，杀鸡不需要牛刀。留一个条件登记：若未来出现规模化图状工件的完整性审计（如 501-dyad 轨迹库的 lineage 引用图、CMS 卡片引用图达到 10⁵ 级节点），"种子抽样 + 有界局部探索 + 倍增分桶 + 预算封顶 ⊥"可作审计脚本骨架替代全图遍历——届时再评估，不预先设计。

6. **成熟度与档位**：**C**——如实登记：顶级图算法结果（两个 20 年开放问题），但与本仓库距离远；只保留"局部探测替代全局计算"的设计模式、单侧错误语义与预算封顶 ⊥ 作方法论参照。不拔高。

7. **风险与不适配**：(i) 全部结果绑定图的组合结构（割/体积/连通性），无学习论对应物；杨柳在本篇的角色也非学习理论视角。(ii) 若强行类比"图 = 快照依赖图"会违反本文件纪律声明第 2 条（表面类比）——已明确拒绝。(iii) 唯一安全携带的是第 4 点的三件套 + 单侧错误方向论；任何更强的主张（如"volvence 审计复杂度可近线性"）没有依据。

---

## 簇级小结

### 净贡献：本簇给 Readable 审计面带来什么

1. **testing ≪ learning 层级成为审计预算的第一性框架**。volvence 已有的纪律——预检零新增 GPU（不变量 7/8 用已收集数据）、formal 才烧大预算（A2 全量 501 dyad ≈ 880 h，主线方案 §6）——在本簇语言里获得统一表述：**验证一个性质的固有代价（testing dimension 级）与学出/跑出该对象的代价（learning/formal 级）是两个不同的量，前者常常低一个多项式量级，且应当先花小钱确认"值得花大钱"**。区间并（O(1/ε⁴) vs Ω(d)）、LTF（√n vs n）、k-分段（与 k 无关 vs 随 k 线性）是三个可引用的分离实例。
2. **testing dimension 视角：预检自身是两假设族判别问题，其样本量可以且应当被论证**。d_S(π, π′) 的形式化把不变量 7 的三个统计量（共模能量、Cohen's d、1-NN）安放为"可分性代理"，并催生一条增量纪律候选：**冻结预检门槛时附功效论证**（给定 n 与门槛效应量，判别功效 ≥ 1−δ）——防止预检在"样本量不足以让两个世界可分"的 regime 空转，即"预检的预检"。A1 的教训（v1 通道把 d 压到 0.315，null 不可解释）正是通道选择抬高有效 testing dimension 的实例。
3. **下界意识与单侧错误语义**。dictator 反例（测试 = 学习 = Ω(log n)）确立"验证便宜需逐性质论证"；passive Ω(√d)/Ω(√k) 下界提示**只读被动预检**（volvence 的常态）的预算下限是 √ 型而非 O(1) 型；LocalEC/所有 tester 的单侧保证方向（发现必真、未发现存疑）与 evaluation spec 的"未触发 ≠ 安全证据"合流为同一条审计公理。

### 对"预检协议样本量"的具体设计启示（量级论证框架）

以下均为设计原则与量级框架，不是任何现有预检已达标的声明；落地须逐条核对假设并随 prereg 冻结数值。

1. **验证/学习预算分离表**（#23 learn-then-validate）：fit 预算 O(complexity/ε · polylog)，validate 预算 O(1/ε · log(1/δ))（一致性通过型）或 O(1/ε² · log(1/δ))（一致率估计型）——**validate 与被验对象复杂度无关**。应用：C2 锚只做 validate（已是契约定位），其量级预算按 (ε, δ, η) 三元组设计，与 gate/reader 参数量脱钩。
2. **heldout 全对的解读规则**（rule of three）：n 个 heldout 样本零错误在 95% 置信下只授权 err ≤ 3/n。S3-前置 heldout 1.0、C3 主判据 readout 预检等一切"满分"结论，其效应量分辨率由 n 反推并写入 prereg；反过来，若 prereg 需要最小可检 ε₀，则 heldout 规模下限 n ≳ 3/ε₀。
3. **对统计的构造成本**（#23 生日配对）：需要 m′ 个 δ-近邻对时，unlabeled 预算 ≈ m′ · O(√(1/δ))（每块 1+⌈2√(1/δ)⌉ 个点自然产出一对）——分辨力预检的 same/diff dyad 对、段结构审计的近邻 turn 对，其池规模都可用此式先验核算。
4. **regime 自觉**：volvence 预检是冻结数据上的被动统计 ⟹ 引用预算直觉一律用 passive 界（√ 型）；只有真的实现"自适应挑样本请人工标注"的协议（当前没有）才允许引用 active 界。
5. **噪声标注的有效样本收缩**：双标注者先估 inter-rater 噪声 η，可检效应按 (1−2η) 收缩 ⟹ pilot 对数按 ε⁻²·(1−2η)⁻² 量级起步；这与 C2 的"先 pilot 双标注者 + 一致性门，再决定扩量"（主线方案 §4.2）的顺序完全一致——一致性门就是在测 η。
6. **负对照的双世界构造**（#14 下界模板 + 不变量 8）：每个预检至少配一个"已知无信号世界"臂（shuffled/swapped 类），否则"可分"可能测的是混杂通道；A1 的 swapped-state 5.93e-05 是该模板已产出的真实收益（发现与标度缺陷独立的第二死因）。

### 供 06 综合文档使用的转化候选表

| 论文 | 轴 | 对象 | 一句话方案 | 档位 |
|---|---|---|---|---|
| #14 | Readable（审计面） | 仪器分辨力预检的形式化 + 功效论证 | 把不变量 7 预检表述为两假设族判别（d_S(π,π′) 视角），prereg 冻结门槛时附"n 下功效 ≥ 1−δ"论证（预检的预检） | A |
| #14 | Readable | 负对照设计模板 | 每个预检配显式"无信号世界"臂（shuffled/swapped），把"无效应"与"仪器盲"拆开——与不变量 8 传导预检同模板，推广到新预检时照抄 | A |
| #14 | Readable/Learnable | 线性 reader 漂移哨兵 | "残差→标签仍线性可读"的监测在 Gaussian 假设下 O(√n·log n) ≪ 重学 Ω(n)；仅作量级直觉，分布假设不满足，不进 prereg 数值 | B |
| #23 | Readable（场景 a/c） | 验证锚/heldout 预算量级 | learn-then-validate 分解：validate 预算 O(1/ε·log(1/δ)) 与对象复杂度无关；heldout=1.0 按 rule of three（err ≤ 3/n）反推最小可检效应；C2 pilot 按 ε⁻²(1−2η)⁻² 定对数 | A |
| #23 | Readable（R3/R4 接口） | belief/β_t 段结构审计 | 近邻 turn 对不一致率（广义 noise sensitivity 的离散化）判"≤k 段分段稳定 vs ε-远"，查询数与 k 无关；SHADOW 诊断脚本级，距离定义随 prereg 冻结 | B |
| #05 | 审计方法论 | 局部探测替代全局计算 | 全局违反 ⟺ 局部见证分解 + 倍增尺度枚举 + 预算封顶显式 ⊥ + 单侧错误方向（发现必真/未发现存疑）；仅作规模化工件审计的脚本设计模式储备 | C |

**跨簇一致性核对**：本表与 00_PAPER_INDEX.md 的档位（14=A，23=A，05=C）一致。本簇不产生任何学习信号或运行时行为改动候选——所有对象都在 prereg 前置面或 evaluation-readout 面，R12（evaluation 只读不回灌）与 R8（快照隔离）在每条候选上保持不变。
