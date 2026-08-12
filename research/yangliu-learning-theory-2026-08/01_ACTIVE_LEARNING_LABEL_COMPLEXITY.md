# 簇 1 · 主动学习标签复杂度（10 篇逐篇深析）

> 研究项目：`research/yangliu-learning-theory-2026-08/`；总索引见 [00_PAPER_INDEX.md](00_PAPER_INDEX.md)。
> 分析对象：杨柳（Liu Yang，CMU ML PhD 2013，导师 Avrim Blum & Jaime Carbonell；Steve Hanneke 长期合作者）"主动学习标签复杂度"簇，正式 10 篇（02/07/09/13/16/17/24/25/30/31）+ 跨簇引用 #34。
> 分析基准日：2026-08-12。所引 Volvence 文档为当日版本：`docs/appendable-readable-learnable-steerable.md`（下称"四轴 charter"）、`docs/moving forward/主线提升方案_2026-08.md`（下称"主线方案"）、`docs/specs/steering-human-anchor.md`、`docs/specs/prediction-error-loop.md`。

---

## 0. 簇定位：本簇 ↔ Learnable 轴 Gate 4 缺口（工作流 C）

**缺口是什么。** 四轴 charter §8 的证据状态表写明：Learnable 轴已证明的是 S3-E（代理迷宫：给定稀而准的终局信用，门控可小样本学会择时）、段信用 v13 retain、C1 的 PE→credit→gate 契约；**未证明**的是"对话域 C3 formal"与"Gate 4 主动学习省标签"。主线方案 §1.1 scorecard 对应行：稀疏主动学习 —— 机制证据 = S3-E；系统缺口 = "Gate 4 `causal not-supported`；companion 域无免费客观信用源 → 工作流 C"。工作流 C 的三个包：C1 主信用 = N+1 substrate 表示 PE（PE-owned，不经 evaluation）；C2 专家标注 = 验证锚（`validation_anchor_only=true`，非学习源）；C3 = S3-E 择时学习向对话域迁移的 prereg。

**本簇提供什么。** "主动学习省标签"这句话在杨柳-Hanneke 的理论体系里被拆成了可判定的命题族：什么噪声模型、什么假设类结构、什么查询协议、什么损失函数之下，主动查询相对被动采样的标签复杂度改善是指数级、多项式级、常数级、还是**不存在**。Gate 4 想要的因果判词（"门控学习在 matched 标注预算下省标签"）恰好需要这套上下界语言来（a）设定可辩护的预期效应量、（b）避免把理论上不可能的改善写进 prereg、（c）把"省标签"主张钉在理论支持的目标形式上。

**杨柳主动学习理论体系的内部结构**（本簇阅读顺序即依此展开）：

```text
disagreement-based 主线（CAL → A2 → RobustCAL）
        │
        ├── minimax 刻画：#09 star number 统一几乎全部既有复杂度度量，
        │   给出六个噪声模型的分布无关上下界（本簇理论核心）
        │
        ├── 噪声条件扩展：#17 uniform classification noise 下的 universal
        │   activizer（任意被动算法可主动化）
        │
        ├── 代理损失：#07 classification-calibrated surrogate 的正确用法
        │   （正结果）↔ #25 凸损失 proper AL 的 minimax 负结果（对偶）
        │
        ├── 查询协议扩展：#13 批量查询（batch/轮次权衡 + 次线性成本）；
        │   #30/#31 多 oracle 成本-可靠性权衡（proactive learning）
        │
        └── 贝叶斯先验：#24 先验弥合"学习 vs 自验证"差距；#16 任意二值
            查询的熵刻画（率失真视角）；#02 参数混合模型（无公开版）
```

这个结构与工作流 C 的对应关系：minimax/噪声支线回答"C3 的择时学习**理论上**能省多少 N+1 PE 结算预算"；代理损失支线回答"gate 的 bounded policy-gradient（代理目标）与 0-1 型择时决策（真目标）之间的标签复杂度换算"；批量/多 oracle 支线回答"C2 的 48-unit 双标注 pilot 与扩量批次、以及免费 PE oracle vs 昂贵专家 oracle 的预算分配"；贝叶斯支线回答"prereg 冻结的结构先验如何替代昂贵的自验证标注"。

## 0.1 分析纪律声明

1. **C2 定位不可越界**：人类专家查询的一切理论落点必须尊重 `steering-human-anchor.md` §1 的 fail-closed 契约（`validation_anchor_only=true`、`learning_use_authorized=false`、`production_promotion_authorized=false`）。因此本文所有"查询策略/标注预算"结合点，近期只允许落在三处：(i) **验证锚标注预算分配与一致性门设计**（C2 的 48-unit pilot、扩量 prereg）；(ii) **内部免费客观标签的查询调度**（N+1 PE heldout 结算的 episode 选择与批调度，信号面完全在 PE owner 内）；(iii) **prereg 的功效/效应量核算**。"专家标注升级为信用源"只能作为远期单独 prereg 的理论准备来写（主线方案 §4.2 升级条件：仅当 C1 免费信用与人类锚不一致且差距 load-bearing），本文按此措辞，不做任何"专家标签直接进学习环路"的转写。
2. **负结果与正结果同等重要**：#25（凸损失）、#09 的下界与 s=∞ 情形、#17 所依赖的"某些噪声模型下 universal activizer 不存在"（Hanneke 2012），都是 Gate 4 判词的诚实边界素材。凡"主动学习一定省标签"在理论上不成立的情形，在簇级小结中显式列出，作为对外表述的禁语清单。
3. **evaluation/judge 禁止作为学习信号**（R12，主线方案 §4.0）：本文一切"标签/oracle/信用"的转写候选，其信号来源只有两类合法项——PE 通道的免费客观标签（N+1 表示 PE，`prediction-error-loop.md` §N+1 表示前向预测）与人工标注通道（仅验证锚）。judge 分数、七日 continuity readout、companion-bench 分数在任何转写中都不出现；若某论文机制天然需要一个"评分 oracle"，本文明确写"不可转写"并给出理由。

---

### 09. Minimax Analysis of Active Learning（JMLR 2015）

**文件**：`papers/09-minimax-analysis-active-learning-jmlr2015.pdf`（112 页；本次深读 §1–§7 主结果、定义与讨论，证明附录 B/C/D 未逐页读）

1. **基本信息**：Steve Hanneke, Liu Yang（字母序，杨柳第二）；JMLR 16 (2015)。作者自评贡献："discovering this new combinatorial complexity measure called the star number that well-characterizes the minimax label complexity of active learning in low-noise regimes and proving that almost all of the complexity measures previously proposed in the active learning literature have worst-case values equal to the star number."
2. **问题设定与核心结果**：pool-based 主动学习（结果同样适用 stream-based selective sampling，论文脚注 3），实例空间 X，假设类 C（VC 维 d），六个噪声模型 RE / BN(β) / TN(a,α) / BC(a,α) / BE(ν) / AG(ν)。核心量是 minimax label complexity `Λ_D(ε,δ)`。**star number**（Definition 2）：最大的 s，使存在点 x_1..x_s 与分类器 h_0,h_1..h_s ∈ C 满足 `DIS({h_0,h_i}) ∩ {x_1..x_s} = {x_i}`（以 h_0 为中心的"星"）。恒有 s ≥ d，且 s/d 之比可为无穷（threshold 类 s=2；interval 类、R^k 线性分类器、axis-aligned rectangles 都是 s=∞）。主定理（保留主项）：
   - Realizable（Thm 3）：`max{min{s,1/ε}, d, Log(min{1/ε,|C|})} ≲ Λ_RE ≲ min{s, d/ε}·Log(1/ε)`。s<∞ ⇒ O(Log(1/ε))（近指数改善）；s=∞ ⇒ Ω(1/ε)（与被动同阶）。
   - Tsybakov（Thm 5）：α ∈ (0,1/2]（高噪声 regime）时 `Λ_TN ≍ a²(1/ε)^{2−2α}·d·polylog`——**与 s 无关**，同 VC 维的所有假设类复杂度相同；α ∈ (1/2,1) 时上界含因子 `min{s/d, 1/(a^{1/α}ε)}^{2α−1}`。任何 α∈(0,1) 下 `Λ_TN = o(M_TN)`（被动 minimax 为 ã(1/ε)^{2−α}·d 阶）：改善因子 s<∞ 时 ≈ ε^{−α}，s=∞ 时 ≈ ε^{−min{α,1−α}}。**这是文献中首个"任意 VC 类在 TN 下主动恒严格优于被动"的结果**。
   - Benign（Thm 7）：`ν²/ε²·(d+Log(1/δ)) + min{s,1/ε} ≲ Λ_BE ≲ (ν²/ε²·d + min{s,d/ε})·polylog`。ν ≥ √ε 时改善因子仅 ≈ 1/ν，且绝对量 ν²/ε² 仍然巨大。
   - 分离结果：TN(a,α) 与 BC(a,α) 的主动复杂度可指数分离（被动下两者等价至 log 因子）。
   - 等价定理（§7，Table 1）：`sup_P θ_P(ε) = s ∧ 1/ε`（disagreement coefficient）、splitting index 的 `sup 1/ρ = s ∧ ⌊1/ε⌋`、extended teaching dimension `XTD(C,m) = s ∧ m`、version space compression `max ĥ_h(U) = s ∧ m`、doubling dimension `∈ [1, O(d)]·log(s∧1/ε)`——几乎全部既有度量的最坏值都是 star number。
3. **核心机制**：上界算法三组件：(i) 把 X 按数据依赖方式划分成 cell，使 f* 在绝大多数 cell 近常值（VC 类 finite approximability 的数据版），对同 cell 多点重复查询以 SPRT 判定多数标签（Kääriäinen 2006 技术的 pool 版）；(ii) **按标签方差分层弃点**——某 cell 判定多数标签耗查询过多则弃掉该点、且阈值随数据量递减，把预算从高噪声区（|η−1/2| 小）挪向低噪声区，这是全文的主要算法创新，也是高噪声 regime 上界与 s 无关的原因；(iii) 对推断出的 f* 标签跑 CAL（disagreement-based），用 Wiener–Hanneke–El-Yaniv 的 version space compression 分析接 star number。下界用 Kääriäinen/Beygelzimer–Dasgupta–Langford 的噪声扰动技术加 Raginsky–Rakhlin 的复杂度嵌入。
4. **思想结合点**：落在 Learnable 轴 Gate 4 的效应量先验。四轴 charter §4.2 的 C1 信用链把择时学习的信号面定为 N+1 表示 PE（稀、准、arm-independent），主线方案 §4.3 的 C3 判定门要求"gain vs noop/always-on/random-gate 的 clustered CI 下界 + worst-seed"。本篇给这类判定门的**预期效应量与标注预算**提供理论模板：对话域信号是高噪声的（A1 v1 formal 的四个 contrast 主效应 0.0058/5.9e-05/0.0074/−0.0004，全部 CI 跨 0；v2 readout d=0.592 也远非可分离），对应本篇的高噪声 regime——那里的定理说改善因子只有 ≈1/ν 或 ε^{−α} 级、绝对标签量 ν²/ε² 仍大，且**假设类结构（star number）不再帮忙**。因此 Gate 4 的 prereg 不应许诺"指数级省标签"（那是 s<∞ + bounded noise/realizable 的专利），只能许诺多项式因子改善，并把"要检验的效应量 × 需要的结算数"用 Thm 5/7 的形式做 power 核算。另一个方向的启示：gate policy 的观测空间是 4 维（`belief_margin, fresh_margin, belief_disagrees_fresh, base_action_entropy`，主线方案 §3.0），有界低维策略族的有效 star number 可控——这是"择时决策目标上省标签在结构上可能"的前提；而若把查询问题定义在原始 residual 空间的线性分类器族上（s=∞），minimax 视角直接判"不省"。
5. **算法结合点**：可转写对象 = **Gate 4 / C3 后继 prereg 的 power 分析段** + **N+1 PE heldout 结算调度**。具体：(a) 把"结算预算 n 与可检出效应 ε 的关系"按 Thm 7 的 `ν²/ε²·d` 主项写进 prereg（信号噪声 ν 用 C3 已收集 trace 的 PE 方差估计，d 用 gate 策略参数量代理），替代拍脑袋的 5-seed×episode 数；(b) 结算调度上，disagreement-based 的"只查 DIS(V)"骨架可实现为：只对"当前 policy 版本与对照 policy 决策不一致的 episode"做昂贵 matched noop/action 双结算——这恰是 PE owner `bind_steering_terminal_prediction_error_decisions(...)` 已支持的 rebinding 面（`prediction-error-loop.md` C3 replay 段），无需新 owner。信号来源核对：全部用 N+1 PE 免费标签，无专家、无 judge，合法。假设匹配度核对：定理假设 i.i.d.（对话流非平稳，不满足——只能做量级参照，不能引用为保证）、二值标签（择时决策是二值，匹配）、假设类已知（gate 策略族已冻结，匹配）。
6. **成熟度与档位**：**A**——高/低噪声 regime 二分 + star number 是 Gate 4 prereg 效应量与预算核算的直接理论模板，且是全簇其余论文的公共语言。
7. **风险与不适配**：minimax 是最坏情形刻画，对话域的实际分布可能远好于最坏（分布相关分析是论文明示的 future work）；上界算法充斥大常数与 polylog、并假设无限无标注池（MSC 语料有限）；i.i.d. 假设在 SHADOW 流上不成立（漂移问题归簇 3）；N+1 PE 是连续回归型信号，本篇的二值标签框架只覆盖"由 PE 差派生的二值择时判断"，不覆盖信用值本身的估计（后者见 #25 的负结果）。

---

### 07. Surrogate Losses in Passive and Active Learning（EJS 2019）

**文件**：`papers/07-surrogate-losses-passive-active-ejs2019.pdf`（62 页；深读 §1–§5 与 §6 的算法/主定理，附录证明未逐页读）

1. **基本信息**：Steve Hanneke, Liu Yang（字母序，杨柳第二）；Electronic Journal of Statistics 13 (2019)。贡献声明："active learning algorithm based on a classification-calibrated surrogate loss function, and analyzing the number of label requests sufficient ... to achieve a given risk under the 0-1 loss."
2. **问题设定与核心结果**：函数类 F，classification-calibrated 代理损失 ℓ（quadratic/exponential/hinge 等），假设 f* = argmin R_ℓ ∈ F。核心桥接量 `Γ_ℓ(ε) = sup{γ : F*(γ;ℓ) ⊆ F*(ε;01)}`，及 Bartlett–Jordan–McAuliffe 的 ψ_ℓ 给出的可算下界 `Ψ_ℓ(ε) = a·ε^α·ψ_ℓ(ε^{1−α}/(2a))`（Condition 3 = Tsybakov 距离条件 Δ(g,f*) ≤ a(er(g)−er(f*))^α 下）。**负结果**（§3.2）：把代理风险优化到足以保证 0-1 超额风险 ≤ ε 的程度，主动学习的 minimax 查询数是 Θ(1/ε)——与被动 ERM_ℓ 同阶（构造：单点 x₀ 上估计 η(x₀) 的 Bernoulli 均值问题）。**正结果**（Thm 8/9，VC subgraph 类）：Algorithm 1 达到 0-1 超额风险 ε 所需标签数约为被动 ERM_ℓ 样本数乘以因子 `θ(aε^α)·a·ε^α`（θ 为 disagreement coefficient）；α<1 时 n = O(θ·ε^α·Ψ_ℓ(ε)^{β−2}·Log(χ_ℓ))，α=β=1 时 O(θ·Log(1/ε)(Log θ + LogLog(1/ε)))。当 ℓ 满足光滑凸性 Condition 2（r_ℓ>2）时（Thm 9），连**无标注**样本需求也乘以 (aθε^α)^{1−β} 因子缩小——甚至全标注数据下也宁可跑 Algorithm 1 而非 ERM_ℓ。
3. **核心机制**：Algorithm 1 = disagreement-based 骨架 + 代理风险版本空间更新。维护 V ⊆ F；对流入的 X_m 仅当 `X_m ∈ DIS(V)`（sign-disagreement 区域）才请求标签；每当 m 翻倍，执行 `V ← {h ∈ V : R_ℓ(h;Q) − inf_{g∈V} R_ℓ(g;Q) ≤ T̂_ℓ(V;Q,m)}`，T̂_ℓ 由局部 Rademacher 复杂度型集中不等式定出，保证 f* 永在 V 中。核心洞察（作者原话的转述）：**只要某点的 sign(f*) 已经确定，就不必再优化该点的代理风险值**——查询只投向符号仍不确定的区域；分析上用 spliced function h_{DIS(V)}（在 DIS(V) 外贴 f*）把代理风险的集中论证局限在采样区域内。
4. **思想结合点**：Learnable 轴的 gate 学习恰是"代理目标 + 0-1 真目标"结构：SteeringGateModule 用 bounded policy-gradient（连续代理）学 STEER/NOOP 二值择时（四轴 charter §4.2），而 C3 判定门只考核择时方向正确性（0-1 型）。本篇的负结果说明"把代理目标优化得足够好"与"用少量标签把决策学对"是两个标签复杂度相差悬殊的问题；正结果给出正确姿势——**按决策符号的不确定性分配查询，而不是按代理残差大小**。这直接映射到 C2 验证锚的预算分配：`steering-human-anchor.md` §3 定义 C1 方向用 deadzone（`relative_mse_improvement > 0.02` 为正、`< −0.02` 为负、中间为中性不入分母），"PE 非中性且人类方向已解析"的 unit 才进 alignment 分母——deadzone 附近正是 sign 不确定的 DIS(V)，把宝贵的双专家标注 unit 优先采样在 |relative_mse_improvement| 靠近 0.02 边界的 decision 上，可以最大化"方向一致性验证"这个二值假设检验的信息量；对 |improvement| 很大、方向早已确定的 decision 重复标注是浪费（对应"sign 已定不再优化"）。
5. **算法结合点**：可转写对象两个。(a) **C2 packet builder 的 unit 采样策略**（`vz-runtime/agent/steering_human_anchor.py` 的上游选样逻辑）：按 |relative_mse_improvement| 与 deadzone 边界的距离分层采样，近边界层加密——这是 Algorithm 1 "查询集中在 DIS(V)"的一次性批版本；注意这改变的是"选哪些 unit 外发盲评"，不改变盲化/一致性门/方向对照契约本身。(b) **N+1 PE heldout 结算调度**：对 gate 决策与 matched-noop 的 PE 差已高置信离开 deadzone 的 episode 类型，降低重复结算频率，把 MPS 预算集中于近边界 episode（信号来源仍全部是 PE 免费标签）。假设匹配度核对：classification-calibrated + f*∈F 是强假设——frozen linear reader 的 margin 是否校准从未验证，因此只借"查询集中于符号不确定区"的结构，不引用其常数级保证；i.i.d. 不满足（同前）；VC subgraph 条件对 4 维观测的有界策略族成立（匹配）。
6. **成熟度与档位**：**A**——deadzone 边界加密采样是可以直接写进 C2 扩量 prereg 的具体方案，负结果部分同时是 Gate 4 措辞的边界素材。
7. **风险与不适配**：T̂_ℓ 的集中阈值依赖对噪声参数或局部 Rademacher 复杂度的估计，48-unit 量级的 pilot 撑不起数据依赖阈值（只能用预注册的固定分层）；近边界加密采样会使 C2 样本相对 decision 总体有偏，alignment rate 的解释必须写明是"边界条件一致率"而非全体一致率——这一措辞必须进 C2 扩量 prereg，否则与 §8 的 0.60 阈值语义冲突。

---

### 17. Activized Learning with Uniform Classification Noise（ICML 2013）

**文件**：`papers/17-activized-learning-uniform-noise-icml2013.pdf` + `papers/supp-activized-icml2013-supplemental.pdf`

1. **基本信息**：Liu Yang, Steve Hanneke（**杨柳一作**，打破字母序）；ICML 2013。贡献声明："for arbitrary VC class, it is possible to transform any passive learning algorithm into an active learning algorithm with strong asymptotic improvements in label complexity under uniform classification noise."
2. **问题设定与核心结果**：uniform classification noise（Angluin–Laird）：存在 h* ∈ C，每个标签以恒定概率 η(D_XY) < 1/2 独立翻转。activizer 定义（Def 4）：主动 meta-algorithm A_a 使得对任意被动算法 A_p 的任意 label complexity Λ_p、任意 nontrivial 分布，`Λ_a(ν + cε, D_XY) = o(Λ_p(ν + ε, D_XY))`——渐近严格快于**任意**非平凡被动上界。主定理：对任意 VC 类，存在 uniform noise 下的 universal activizer。背景负结果（Hanneke 2012，本篇的动机）：存在噪声模型使某些 VC 类**不存在** universal activizer——所以"任意噪声下被动都能免费主动化"是假的，本篇划出的正域是 uniform noise。
3. **核心机制**：三个 subroutine 的流水线。**Subroutine 1**（shatterable-set 去噪，继承 Hanneke 2012）：先用 n/2 预算的随机标签把 C 剪成经验风险近优的 V；再按"V 能 shatter S∪{X_m} 的条件概率是否 ≥1/2"决定请求真标签（入 Q_k）还是推断标签 ŷ（入 L_k）；Lemma 1 保证存在 k*，使 |L_{k*}∪Q_{k*}| = ω(n) 且 L_{k*} 上推断标签全部等于 h*（**去噪标签**）。**Subroutine 2**（re-noising，本篇的独创贡献）：去噪样本不能直接喂被动算法——很多被动算法（uniform noise 下的线性分类器算法、logistic regression 等）的行为依赖噪声统计，**选择性去噪反而使其变差**；于是先用 s=n 个标签粗估 η̂（Hoeffding 给 |η̂−η| ≤ c·n^{−3/8}），再在宽 2n^{−5/16} 的置信区间内以步长 2n^{−17/16} 网格枚举 n^{3/4} 个候选 η_j，对每个 η_j 把去噪标签以概率 η_j 独立重新翻转，得到 n^{3/4}+1 个数据集 R_j；似然比论证（Lemma 2）保证最优 j* 的 R_{j*} 与真实 i.i.d. 样本 Z_m 在被动算法眼里几乎同分布：`E[min_j er(A_p(R_j)) − ν] ≤ (1+o(1))·E[er(A_p(Z_m)) − ν] + (1+o(1))e^{−n^{1/4}}`。**Subroutine 3**（tournament）：对全部候选分类器做成对比较——查询两者不一致点的标签、错得少者晋级——从中选出与最优者可比的胜者。
4. **思想结合点**：本篇给 Learnable 轴两条结构性教训。第一，**"信号分布保真"优先于"逐点更准"**：re-noising 的存在本身说明，给下游学习器提供比部署分布"更干净"的训练信号可能适得其反。映射到 C1 信用链（四轴 charter §4.2）：如果为了"提高信用质量"而对 N+1 PE 结算做选择性过滤（比如只保留高置信、低方差的 settlement 入账），gate 的 policy-gradient 所面对的信用分布就会偏离部署时的分布——C1 现行设计的 arm-independent、全量入账（同 (episode, head) 只去重不筛选）恰好符合本篇的教训，这为该设计提供了理论辩护，也为将来任何"信用清洗"提案设置了反对论据。第二，activizer 的**存在域是噪声模型的函数**：uniform noise 有 universal activizer，某些噪声模型没有（Hanneke 2012）；对话域的噪声结构未知且几乎必然非 uniform，因此"被动基线可被免费主动化"不能作为 Gate 4 的默认预期，只能作为受限假设下的参照。
5. **算法结合点**：tournament（Subroutine 3）结构上对应 C3/B3 的 **candidate bundle 选择**：multi-restart + 训练侧选择已是 S3-E/C3 的预注册机制（主线方案 §4.3），tournament 提供的增量是"用**成对不一致点上的少量标签**在候选间做淘汰"。转写时的信号来源核对是关键：成对比较的裁决标签**只能用 N+1 PE 免费标签**（对两个候选 gate 决策不一致的 episode 做 matched 双结算、按 PE 差裁决）；**不得**用专家标注做 tournament 裁决——那会让人类标注直接选择 policy，等于绕过 credit owner 成为学习/选择源，违反 `steering-human-anchor.md` §1 的 `learning_use_authorized=false` 与 §8 的"一致率不是 C3 admission/B3 promotion 输入"。假设匹配度核对：uniform noise（对话域不满足，PE 噪声是异方差的）；需要可重复查询同分布标签（PE rebinding 可对新 lineage 重放冻结 mismatch，形式上满足"再问一次"，但不产生独立噪声样本——独立性假设不满足，tournament 的置信保证要打折）；VC 类（gate 策略族满足）。
6. **成熟度与档位**：**B**——两条思想教训（分布保真、activizer 存在域）直接可用于设计评审与 prereg 措辞；tournament 转写需要先解决"PE 重放不产生独立样本"的差距，是思想级参照而非即插即用。
7. **风险与不适配**：全篇是渐近结果（o(·)、ω(·)、常数 c 可任意接近 1 但未量化），有限样本无保证；meta-algorithm 需要海量无标注数据与多次重训被动算法（对应我们反复重放 SHADOW trace + 重训 gate，MPS 预算敏感）；uniform noise 假设强到对话域基本不可能成立——本篇价值在结构与边界，不在数值。

---

### 25. Negative Results for Active Learning with Convex Losses（AISTATS 2010）

**文件**：`papers/25-negative-results-convex-losses-aistats2010.pdf`

1. **基本信息**：Steve Hanneke, Liu Yang（字母序，杨柳第二）；AISTATS 2010。贡献声明："even under bounded noise constraints, the minimax rates for proper active learning are often no better than passive learning."
2. **问题设定与核心结果**：proper 主动学习（必须输出 f̂_n ∈ F），目标是凸损失 ℓ 的超额风险 E[R(f̂_n) − R*]。**Theorem 1**（严格凸损失）：ℓ 二次可微、ℓ>0、ℓ′<0、ℓ″>0 处处连续，F 在某点 x₀ 的取值覆盖 [1/2,1]，则存在边际分布、噪声界 η<1/2 与常数 c，使**任意**主动学习算法在某 η-bounded-noise 标签分布下 `E[R(f̂_n) − R*] > c/n`——与被动学习的 c′/n 率（Bartlett–Jordan–McAuliffe）同阶，主动零改善。证明是干净的归约：P 集中于单点 x₀，条件风险极小点由 `ν·ℓ′(−y) = (1−ν)·ℓ′(y)` 给出 ν ↔ y* 的连续双射 φ，于是"学 f̂_n(x₀)"等价于"估计 Bernoulli 均值 ν"，而 Bernoulli 均值估计的 minimax 平方误差 > c/n 与查询自适应性无关（所有查询都在 x₀，主动性无从发挥）；强凸性把 (ŷ−y*)² 下方联系到超额风险。**Theorem 2**（一般凸损失 + 一维常斜率线性类 F = {x−t}）：即使 ℓ 只是非增连续凸（hinge 型也覆盖），在 [0,4z] 均匀边际、远端区域 (2z,∞) 以概率 ν 翻转的 1/4-bounded-noise 分布族上，同样 `E[R(f̂_n) − R*] > c/n`。
3. **核心机制**：两个定理共享同一直觉——**远处的噪声点拖拽凸风险的最优解**：凸损失下，距离决策边界很远的点的噪声幅度持续影响 argmin（hinge/exp/logistic 都如此），学习器必须**估计噪声幅度**才能定位最优解，而"估计一个 Bernoulli 参数的幅度"是主动查询无法加速的任务（每次查询就是一次 Bernoulli 抽样）；0-1 损失则相反——一旦某点的最优符号确定即可永久忽略该点，查询得以向决策边界局域化。作者在 §6 强调解读边界：这不排除"用凸代理设计主动算法去优化 0-1 损失"（正是 #07 后来做的），但排除"以凸风险本身为目标的主动改善"，且警告为代理损失的最坏情形做优化的算法会主动去搜噪声区、自毁局域化。
4. **思想结合点**：这是 Gate 4 判词措辞的**第一硬边界**。Volvence 的信用是连续标量（C1 primary = `clip((noop_mse − action_mse)/max(...), −1, 1)`，四轴 charter §4.2），如果把 Gate 4 的"省标签"目标形式化为"用更少结算把**信用值/PE 改善量**估计到给定精度"——那是凸回归型目标，本篇判定主动调度**不省预算**（估计量的方差由 Bernoulli/回归噪声决定，与查哪个点的自适应性无关）。省标签主张唯一能挂靠的形式是 **0-1 型择时决策质量**（该扳/不该扳的方向正确性），因为只有那里"符号确定即可忽略"的局域化机制才成立。这与 #07 的正结果构成精确的对偶：同一套 volvence 对象（gate + PE 信用），目标写成"决策方向"则理论支持省标签，写成"信用估计"则理论否定。主线方案 §4.3 的 C3 判定门恰好已是方向型（gain 的 CI 下界 + 门控选择性），本篇的作用是**禁止未来的 prereg 把判据滑向信用拟合误差**。
5. **算法结合点**：负结果本身不可转写为算法，转写对象是 **prereg 的判据措辞约束**：(a) Gate 4 后继 prereg 中，主判据必须是二值决策面上的量（择时方向正确率、matched-budget 对照的方向性 gain），任何"用更少标注把 relative_mse_improvement 估准"的表述都要在设计评审时引用本篇打回；(b) C2 验证锚侧同理——`steering-human-anchor.md` §3 的标签语义（双专家方向一致性 + deadzone）已是二值/三值判决而非打分回归，本篇支持维持该设计、反对将来改为"专家给 improvement 连续打分再对齐"的提案（连续对齐的样本复杂度是 1/n 率不可加速）。信号来源核对：不涉及新信号。假设匹配度核对：proper 限制（gate 只能输出策略族内策略——满足）；bounded noise（对话域未验证，但负结果在更弱假设下只会更负，方向安全）。
6. **成熟度与档位**：**A**——作为诚实边界素材直接进入四轴整合清单：它划定了"省标签"主张的合法目标形式，是防止 Gate 4 判词被写歪的最便宜保险。
7. **风险与不适配**：定理是 minimax 的——特定良性分布下凸目标的主动改善仍可能存在（论文自己留了口子）；构造依赖"远端噪声区有非零质量"，若对话域的 PE 噪声天然集中在决策边界附近，实际情形可能温和于最坏情形；但 prereg 措辞应按最坏情形防御，这正是负结果的用法。

---

### 24. The Sample Complexity of Self-Verifying Bayesian Active Learning（AISTATS 2011）

**文件**：`papers/24-self-verifying-bayesian-al-aistats2011.pdf`

1. **基本信息**：Liu Yang, Steve Hanneke, Jaime Carbonell（**杨柳一作**）；AISTATS 2011。贡献声明："access to a prior distribution over target functions can dramatically improve the sample complexity of self-terminating active learning algorithms, so that it is always better than the known results for prior-dependent passive learning."
2. **问题设定与核心结果**：noise-free 二分类，目标 h* ~ π（已知先验），算法输入 (π, D, ε)，**自终止**（自己决定何时停并保证 E[er(ĥ)] ≤ ε）。已知背景：prior-dependent 被动学习是 Θ(1/ε)（Haussler–Kearns–Schapire 的 d/ε 上界，且 threshold+均匀先验给出 Ω(1/ε) 下界）；prior-independent 的自验证主动学习在 interval 类上**做不到** o(1/ε)（Balcan–Hanneke–Vaughan 2010：学得快 ≠ 能便宜地验证自己学好了——学习与验证的标签复杂度存在本质差距）。**主定理（Thm 1）**：任意 VC 类 C、任意 D 与 π，存在正确的 prior-dependent 自终止算法使 `SC(ε,D,π) = o(1/ε)`——先验访问**总是**弥合学习-验证差距、严格超越先验相关被动学习。热身例（intervals，D=Uniform[0,1]，任意先验）：按二进网格 1/2, 1/4, 3/4, 1/8,… 查询直到（情形 1）遇到正点则左右二分搜索 log₂(2/ε) 次定端点，或（情形 2）后验期望宽度 `E[w(h*)|V] ≤ ε` 则直接返回全负分类器；关键是情形 2 的停止条件只有拿着先验才能算。
3. **核心机制**：一般证明是优雅的归约。Lemma 2（Hanneke 2009 的 budget-based 结果）：存在 prior-independent 的预算型算法 A_a 与函数 R(n;f,D) ≤ c/n 且逐目标 o(1/n)，使 `E[er(ĥ_n)|h*] ≤ R(n;h*,D)`——**学习本身**不需要先验就能逐目标 o(1/n)。先验的作用只在**验证/停止**：定义 `n_ε = min{n : E[er(ĥ_n)] ≤ ε}`，其中期望对 h* ~ π 遍历——这个量只有知道 π 才能计算；自终止算法就是"跑 A_a(n_ε) 然后停"。Lemma 1 的技术处理把逐目标 o(1/n) 汇聚成先验平均 o(1/n)（对角化构造 φ̄_n 使 P(φ_n(h*) > φ̄_n) → 0），于是 E[er(ĥ_n)] = o(1/n)，n_ε = o(1/ε)。**一句话：学习不贵，验证才贵；先验把验证的成本转移给了结构知识。**
4. **思想结合点**：Learnable 轴上处处是 self-verification 问题：C3 admission（择时学习是否成立）、B3 promotion（SHADOW→ACTIVE 是否可晋升）、乃至 C2 的"要多少 unit 才够下结论"。本篇的分解——**学习的标签复杂度 vs 验证的标签复杂度是两个量，先验/结构知识可以只补验证那一半**——精确解释了 volvence 证据体系的经济学：为什么 C2 只需要 48 units（`steering-human-anchor.md` §7）就敢谈方向一致性？因为它验证的不是"gate 学到的 policy 好"这个大假设（那需要 Ω(1/ε) 级标注），而只是"C1 免费信用与人类判断方向一致"这个一维 Bernoulli 假设；"policy 学得好"的验证被转移给了 prereg 冻结的**结构先验**——S3-E 已在代理域证明"给定稀而准信用该结构能学会择时"（四轴 charter §4.2 明言 C1 性质对齐 S3-E 证明前提），C3 只需检验信用面迁移。这正是 n_ε 型设计：判定门、seeds、预算在看到数据前由结构知识冻结（主线方案 §0 不变量 3），运行时不再消耗标注去"边看边验证"。
5. **算法结合点**：转写对象 = **C3/B3 的停止与预算条款**。(a) 正向转写：C3 prereg 的 episode 预算与 multi-restart 数应显式引用 S3-E 的样本量作为"先验 n_ε 估计"（跨域先验），并在 prereg 文本里声明"该预算是结构先验下的 n_ε，运行中不得因看到中间结果而延长"——对应 `prediction-error-loop.md` 控制面已有的"中间 checkpoint 禁止用于换 seed/选容量/产生 effect verdict"；(b) 反向约束：任何 data-dependent stopping（跑到显著为止）都等价于没有先验时的自验证，其标注/结算成本会回到 Ω(1/ε) 量级且引入选择偏差——用本篇作为反对"adaptive formal"提案的引用。信号来源核对：不新增信号。假设匹配度核对：known prior π 在 companion 域**不存在**——我们只有"结构先验"（S3-E 的机制证明 + 冻结判定门），定理的数值保证不迁移，迁移的是"先验补验证"的设计模式；noise-free 假设不满足（对话域高噪声），故 o(1/ε) 的具体率不可引用。
6. **成熟度与档位**：**B**——不提供可跑的算法，但"学习/验证分解 + 先验补验证"是 volvence prereg 纪律的理论根，值得写进 06 综合文档的方法论一节。
7. **风险与不适配**：把 S3-E 当"先验"是类比而非概率意义上的 π——若代理域与对话域的结构差距大（正是 C3 要检验的），先验就是错的，n_ε 型预算会不足或过量；本篇 noise-free + known D 的设定与对话域距离远；预算不足时的正确动作是按主线方案 §4.3 退出条件如实封存，而不是引用本篇加预算。

---

### 16. Bayesian Active Learning Using Arbitrary Binary Valued Queries（ALT 2010）

**文件**：`papers/16-bayesian-al-binary-queries-alt2010.pdf`（信息论期刊稿 `papers/supp-lossy-coding-journal-manuscript.pdf` 为同一结果的率失真表述，本次以会议版为准）

1. **基本信息**：Liu Yang, Steve Hanneke, Jaime Carbonell（**杨柳一作**）；ALT 2010。贡献声明："derived bounds on the expected number of queries required to achieve a specified expected risk for a general Bayesian active learning setting in which the learner can ask arbitrary yes/no questions."
2. **问题设定与核心结果**：抽象设定——伪度量空间 (C*, ρ)，先验 π 支撑在 C ⊆ C* 上，目标 h* ~ π；学习者可问**任意**是/否问题，求 `QueryComplexity(ε)` = 保证 `E[ρ(ĥ,h*)] ≤ ε` 的最小期望查询数。关键等价：确定性查询算法 ⇔ prefix-free 二进制码（决策树左 0 右 1，叶存 ĥ）——于是问题就是**率失真**：设计有损码使期望码长最小、期望失真 ≤ ε。设 Y(ε) 为极大 ε-packing，P(ε) 为其诱导的 Voronoi 划分，H(S) = −Σ π(S)log₂π(S) 为划分熵，d 为 doubling dimension。**主定理（Thm 1）**：d < ∞、ρ̄ < ∞ 时，`H(P(ε·log₂(ρ̄/ε))) − O(d) ≤ QueryComplexity(ε) ≤ H(P(ε)) + 1`。上界即对 Voronoi cell 的 Huffman 码；下界（主要贡献）需处理 cell 无分离间隔时失真预算的泄漏。最坏先验（均匀）下退回 Kulkarni–Mitter–Tsitsiklis 的 log M(ε)（packing 数对数）；信息丰富的先验使熵远小于 log M(ε)——**先验的价值被熵精确计价**。
3. **核心机制**：上界：Huffman 码对 P_ε(h*) 编码，期望码长 ≤ H+1，解码到 cell 代表点 Y_ε(h*)，极大 packing 同时是 ε-cover 故失真 ≤ ε。下界：若算法期望查询数 < H − O(d)，则由 Fano/计数式论证，回答序列的信息量不足以把后验集中到直径 ε·log₂(ρ̄/ε) 的 cell 内，doubling dimension 控制"跨 cell 蹭精度"的余地（Gupta–Krauthgamer–Lee 的 packing 计数 |{h′∈Y(γ): ρ(h′,h) ≤ δ}| ≤ (4δ/γ)^d）。
4. **思想结合点**：这是**标注预算的信息论地板**——任何二值查询协议（标签请求、A/B 偏好、是/否问句）的期望次数下界是目标不确定性的熵。用它审计 C2 的量纲（`steering-human-anchor.md` §7）：pilot = 48 units × 2 raters = 96 assignments，每个 assignment 产出 1 个强制偏好（≤1 bit）+ 3 个 1–5 分维度（≤ 3·log₂5 ≈ 7 bits，且维度间强相关、有效信息远低于上限）。若妄图用这点信息**学习** gate policy（策略空间的先验熵即使按 4 维观测的粗离散化也远超 10³ bits），信息论直接判死刑；而 C2 实际要判定的只是"C1 方向与人类方向一致率 ≥ / < 0.60"——一个一维 Bernoulli 参数的区间判定，几十个有效 bit 即够。**熵视角把"C2 只能做验证锚、不能做学习源"从契约约束升格为信息论必然**：不是我们不许它学，是它的信息量根本学不动，任何"就地用 C2 数据改 gate"的提案在量纲上就是虚假承诺。
5. **算法结合点**：转写对象 = **C2 扩量 prereg 的功效核算**（§7 预留的 120–240 units 独立 power/budget prereg）。做法：把待判假设写成显式集合（如"alignment ∈ {≥0.75, 0.60–0.75, <0.60}"三格），按先验不确定性算区分它们所需的期望信息量，除以每 unit 的有效信息（用 pilot 的 rater 相关性估计），得到 unit 数下界——替代纯 rule-of-thumb 的 120–240。信号来源核对：只涉及人类验证锚自身的预算核算，不触学习环路。假设匹配度核对：known prior（对 Bernoulli 参数可用无信息先验，成立）；任意 yes/no 查询（实际协议受限于盲评 rubric，实际效率低于理论最优——所以熵下界是**乐观下界**，实际预算须上浮）；doubling dimension 有限（一维参数显然成立）。
6. **成熟度与档位**：**B**——一个干净的量纲审计工具与"验证锚定位的信息论论证"，不产生新算法。
7. **风险与不适配**：任意查询假设远强于真实标注协议（真实查询不能任意设计、rater 有噪声、盲化约束进一步降低每查询信息量），熵下界与真实预算之间的常数差距可能很大；期望复杂度（平均情形）而非高概率保证，prereg 的判定门若要求 worst-case 置信还需再加系数；把策略空间熵的估计写实需要对策略族做离散化，粗糙离散化会高估学习不可行性的余量（虽然方向不变）。

---

### 13. Buy-in-Bulk Active Learning（NIPS 2013）

**文件**：`papers/13-buy-in-bulk-active-learning-nips2013.pdf`（完整证明见 `papers/supp-buy-in-bulk-techreport-cmu-ml-12-110.pdf`）

1. **基本信息**：Liu Yang, Jaime Carbonell（**杨柳一作**）；NIPS 2013。贡献声明："the label complexity bound of active learning algorithms that request labels in a given number of batches, as well as the tradeoff between the total number of queries and the number of rounds allowed."
2. **问题设定与核心结果**：动机是标注的批量折扣（实验 setup 成本、并行标注者、序贯轮次的时间延迟），把全序贯主动学习推广为 k 批模式。**k-batch CAL**（realizable，Thm 3.1）：把预算 n 均分 k 批，每批只收 DIS(V) 内的点、批末一次性请求标签并收缩版本空间；标签复杂度 `λ(ε,δ) = O(k·ε^{−1/k}·θ(ε)^{1−1/k}·(d·log(1/ε) + log(1/δ)))`。该式在 k 上精确插值：k=1 退回被动 ERM 的 ~d/ε；k = log(1/ε) 时 ε^{−1/k} = e，恢复全序贯 CAL 的 `O(θ(d log θ + log(log(1/ε)/δ))·log(1/ε))`；**k=2 已把 1/ε 压到 1/√ε**（θ 有界时）。**k-batch Robust CAL**（Tsybakov，Thm 4.2）：令 β = α/(2−α)，β̄ = Σ_{i=0}^{k−1} β^i，则 `λ = O(k·(1/ε)^{(2−α)/β̄}·(c₂θ(c₂ε^α))^{1−β^{k−1}/β̄}·(d log(d/ε) + log(kd/(δε)))^{(1+β·β̄−β^k)/β̄})`——k=1 匹配被动 minimax（至 log 因子），k→大 收敛到全序贯 RobustCAL。**Cost-Adaptive CAL**（次线性成本 c(m)，Thm 5.1）：总成本 `O(c(θ(ε)(d log θ(ε) + log(log(1/ε)/δ)))·log(1/ε))`——全序贯 CAL 标签复杂度的**主因子整个塞进了 c(·) 的括号内**；批大小按几何倍增自适应（c(q′−q) ≥ 2c(q)），版本空间不确定质量减半即开新轮。Tsybakov 版本总成本 `O(c(θ·c₂²·ε^{2α−2}·d·polylog)·log(1/ε))`。
3. **核心机制**：k-batch 的归纳论证——批 b 的 M 个点是 DIS(V_{b−1}) 上的条件 i.i.d.，PAC 界给 `V_b ⊆ B(h*, c(d log(M/d)+log(k/δ))/(M·P(DIS(V_{b−1}))))`，而 `P(DIS(V_{b−1})) ≤ θ(ε)·max_h er(h)`，迭代 k 次得误差 ~ `(c(d log M + log(k/δ))/M)^k·θ^{k−1}` 的几何压缩；批间自适应是全部收益来源（同批内无自适应）。Robust 版把"零错误一致"换成 `(er(h;L) − min_g er(g;L))·⌊n/k⌋/(m−m_b) ≤ E_{m−m_b}` 的 Massart–Nédélec 型阈值收缩，负二项分布 + Chernoff 控制两批间隔的无标注消耗；噪声参数可用数据依赖的局部 Rademacher 阈值 Ê_m 替代（无须先知 α）。
4. **思想结合点**：volvence 的标注与结算天然是批模式，本篇是三条预算结构的直接理论参照。(i) **C2 两阶段设计**：`steering-human-anchor.md` §7 的结构是 pilot 48 units → 一致性门 → 独立 prereg 决定是否扩量 120–240 units，这正是 k=2 的批设计；Thm 3.1 的 ε^{−1/k} 因子给出理论理由——**两轮自适应远优于一次性大批**（k=1→k=2 从 1/ε 到 1/√ε 的跃迁是全部 k 中最大的边际收益），且第二批的选样范围应由第一批后的"版本空间"（rubric 是否过门、哪类 unit 分歧集中）决定，与现行"pilot 过一致性门才谈扩量"的契约同构。(ii) **N+1 PE 结算的批调度**：每次 heldout 结算走 MPS 且全 formal 共享互斥锁（主线方案 §6），批的 setup 成本显著、c(m) 次线性（batch forward 均摊加载成本）——Cost-Adaptive CAL 的"主复杂度因子进 c(·)"说明按版本空间收缩节奏几何倍增批大小的调度在总成本上接近全序贯的信息效率。(iii) 对 Gate 4 判词：省标签的量化必须声明轮次预算 k，"matched 标注预算"对照要 matched 的不只是标签数还有轮数。
5. **算法结合点**：可转写对象 = **C2 扩量 prereg 的批结构条款** + **C3 后继结算调度**。具体：(a) 扩量 prereg 用 Thm 3.1/4.2 的形式论证"48 + 192 两批"相对"一次 240"的样本效率优势，并把第二批的分层（按 pilot 分歧模式）写成预注册规则；(b) 结算调度按 Cost-Adaptive CAL 骨架：批大小几何倍增、"版本空间减半"翻译为"gate 候选集在 PE 差裁决下的存活率减半"。信号来源核对：(a) 是验证锚预算分配（合法），(b) 全部 PE 免费标签（合法）。假设匹配度核对：realizable/Tsybakov + θ 有界（对话域未知，量级参照）；批内条件 i.i.d.（SHADOW trace 重放近似满足——trace 冻结后 episode 可交换）；预算均分 k 批是分析简化，实际几何批更优（论文 §5 自己给出）。
6. **成熟度与档位**：**A**——k 批插值公式与 cost-adaptive 结构是可直接引用进 C2 扩量 prereg 与结算调度设计的闭式参照。
7. **风险与不适配**：均分批的 Thm 3.1/4.2 未证 minimax 最优（作者自认 RobustCAL 本身非最优）；分析要求批间版本空间可显式维护（gate 候选集小、可行；若将来策略族连续化则需 cover 近似）；θ(ε) 在对话域不可估——引用时只能做敏感性分析而非点估计；延迟成本（患者恶化的类比 = 用户体验窗口）在 volvence 是 SHADOW 离线，暂不 binding。

---

### 30. Cost Complexity of Proactive Learning via a Reduction to Realizable Active Learning（CMU-ML-09-113, 2009）

**文件**：`papers/30-proactive-learning-cost-complexity-cmu-ml-09-113.pdf`

1. **基本信息**：Liu Yang, Jaime Carbonell（**杨柳一作**）；CMU 技术报告 CMU-ML-09-113，2009-11。贡献声明："general approach for Proactive Learning that addresses the cost vs. reliability tradeoff for oracle and instance selection, and two types of sequential hypothesis tests that estimate the label of a given query from the noisy replies of different oracles with varying reliabilities and costs."（proactive learning 的首篇理论工作）
2. **问题设定与核心结果**：n 个**非持久**oracle（同一 x 重复问可得独立答案），oracle j 成本 c_j、未知噪声界 α_j < 1/2（bounded rate class noise：`P(y ≠ f*(x)|x) ≤ α_j`）；目标：以高概率 1−δ 学到 `P(f(x) ≠ f*(x)) ≤ ε`，最小化**总成本**（cost complexity 取代 sample complexity）。**ProAL 归约**：取任意 realizable 主动学习算法 A（样本复杂度 N(ε,δ)），A 选点、SeqHTRoutine 从多 oracle 噪声回答中判定"真标签"回喂 A；第 i 次调用的置信参数取 δ′_i = δ/(4i²)（Σ1/(2i²) = π²/12 < 1 保证联合失败率 ≤ δ/2）。**ST1**（n-threaded SeqHT）：每次选使 `(|z_j|+1)·c_j` 最小的 oracle 追加查询（等花费扩张），每个 oracle 独立跑 Hoeffding 置信区间 `p_j ± √(2ln(4n|z_j|²/δ′)/|z_j|)`，任一区间排除 0 即停。**Theorem 1**：tProAL 总成本 `≤ min_j [64·n·c_j/(1/2−α_j)²·ln(64nN(ε,δ/2)/(δ(1/2−α_j)))]·N(ε,δ/2)`——**不知道哪个 oracle 最优，只比"先知最优 oracle"多花 2n 因子**。**ST2**（central pool）：把所有 oracle 的样本并入单一池跑一个 SeqHT；**Theorem 2** 给出调和平均形式的成本 `~ (Σ_j 1/c_j)/(Σ_j β_j/c_j)²·ln(...) + ...`（β_j = 1−2α_j）——多个中等 oracle 拼起来可优于单最优 oracle（§5 进一步给出双跑 ST1/ST2 取先停者、代价 ≤ 2 倍的聚合过程）。SeqHT 的 1/√n 率最优性（Wald）说明置信区间形式无本质改进空间（至多省 log(1/(1−2α)) 因子）。
3. **核心机制**：把"噪声多 oracle"问题**归约**为 realizable 主动学习 + 逐点序贯去噪；成本分析的支点是选 oracle 规则维持的不变量 `max_j c_j|z_j| ≤ min_j c_j(|z_j|+1) ≤ 2·MinCost`（MinCost = min_j M_j c_j，M_j 由 (1/2−α_j) 与 Hoeffding 半径的交点解出，用 `m ≤ w + u·ln(vm) ⇒ m ≤ 2w + 2u·ln(uv)` 化简）。
4. **思想结合点**：volvence 恰有一个两 oracle 结构：**免费噪声 oracle** = N+1 PE 结算（c ≈ 0（GPU 摊销），"噪声率"= PE 信号对真实"该扳"方向的失配率，未知）与**昂贵可靠 oracle** = 双专家标注（≈10 min/assignment、pilot 全程 ≤ 40 person-hours，`steering-human-anchor.md` §7）。主线方案 §4.0–§4.2 的架构决策"主信用 = 免费 PE、专家只做验证锚"在本篇框架里是 oracle 选择问题的一个解：Theorem 1 的判据 `c_j/(1/2−α_j)²` 说明**只要免费 oracle 的噪声率 bounded away from 1/2，成本近零使它永远是主 oracle**；昂贵 oracle 值得付费当且仅当免费 oracle 的 α_j → 1/2（信号失效）。这精确对应主线方案 §7 的 R-C1（"N+1 PE 对 steering 动作不敏感"= α → 1/2 ⇒ 封存该信号面、回到 C2 升级路径讨论）与 §4.2 的升级条件（"仅当 C1 免费信用与人类锚不一致且差距 load-bearing"才考虑专家标注经 credit owner 升级——届时单独 prereg）。**本篇给那个远期 prereg 预置了理论骨架**：升级判据可写成两 oracle 的 `c_j/(1/2−α_j)²` 显式比较，其中免费 oracle 的 (1/2−α) 由 C2 pilot 的方向一致率估计（一致率 0.60 阈值 ↔ α ≈ 0.40 的粗界）。
5. **算法结合点**：近期可转写的只有验证锚侧的**一致性门设计视角**：C2 的双专家一致（M=2 的 agreement test）是 SeqHT 在预算 2 处截断的特例；ST1 的自适应停止（CI 排除 0 才停）对应"分歧 unit 若允许第三标注者何时值得加标"——本篇给出加标的边际价值公式（Hoeffding 半径收缩率），可用于扩量 prereg 里"分歧 unit 处理"条款的论证；但现行契约把分歧 unit 标为不确定、不强行补成正负例（`steering-human-anchor.md` §3），是比 SeqHT 更保守的选择，本篇视角支持在扩量版本中论证"预算允许时三标注者裁决"的净收益。**不可转写部分**：ProAL 的"SeqHT 去噪后当 realizable 标签回喂学习器"若把专家标注当 oracle 之一喂给 gate 学习——那就是专家标签进学习环路，违反 C2 契约；该路径必须等升级 prereg。非持久 oracle 假设（同 x 独立重询）对专家不成立（同一 unit 双标注是 persistent 双样本，见 #31）、对 PE 结算也不成立（同 (episode,head) 只入账一次，rebinding 不产生独立噪声）。
6. **成熟度与档位**：**B**——多 oracle 成本框架是"C1 主信用 + C2 验证锚"架构的事后理论化与远期升级 prereg 的骨架来源；近期无直接可跑的转写。
7. **风险与不适配**：bounded noise + 非持久 oracle 两条假设在 volvence 都不严格成立；成本模型是每查询计价，专家标注的真实成本结构含固定 setup（校准 2 小时/人）与批量折扣——与 #13 的成本函数框架组合使用更贴切；报告是技术报告未经同行评审（正确性论证自足但常数粗糙）。

---

### 31. Adaptive Proactive Learning with Cost-Reliability Tradeoff（CMU-ML-09-114, 2009）

**文件**：`papers/31-adaptive-proactive-learning-cmu-ml-09-114.pdf`

1. **基本信息**：Liu Yang, Jaime Carbonell（**杨柳一作**）；CMU 技术报告 CMU-ML-09-114，2009-12。贡献声明："theoretical framework for proactive learning, and ... a meta-procedure for the active learning problem with multiple persistent oracles under arbitrary noise."
2. **问题设定与核心结果**：与 #30 互补的另一极——n 个**持久** oracle（同 x 重复问答案不变，重询无效），噪声**任意**（不 bounded），Assumption 1：各 oracle 的错误指示 `1(O_i(X) ≠ f*(X))` 相互独立。于是唯一出路是**oracle 集成**：选子集 S 做（加权）多数投票。多数票错误率 `er_maj ≤ exp(−2|S|(1/2−ε̄_S)²)`（Hoeffding + 独立性）；加权版最优权重由极小化该界得 `w_i ∝ (1/2 − ε_i)`（噪声率 1/2 的 oracle 权重为 0）。**AdaProAL**：主动学习进程按精度阶梯 ε_t = 2^{−t} 前进，每阶段调用 OrSelRoutine(ε_t) 重选 ensemble S_t——早期粗精度用小/廉价 ensemble，后期加大。**Assumption 3**（cost-reliability 幂律）：∃β,γ>0 使 `c_i·ε_i^γ ≤ β`；在"成本 c 的 oracle 噪声率 = (β/c)^{1/γ}"取紧时，解 `min c·M s.t. (β/c)^{1/γ} = 1/2 − √(ln(2/ε)/(2M))` 得闭式最优：`c* = β·2^γ·(1+2/γ)^γ`（**与 ε 无关**）、`M* = ⌈2·ln(2/ε)·(1+γ/2)²⌉（随精度只对数增长）`。Theorem 1：以 A² 为基算法的总成本 `Õ(θ²(d+log(1/δ)))·c*·M*`。§4.2：自适应 ensemble 相对固定 ensemble 省常数因子（阶梯求和 Σ_t ln(1/ε_t) ≈ log²(16θe/ε)/(4e²) vs 固定 ln(2/ε)·log(1/ε)），与 Donmez–Carbonell 2008 的实验（约省一半成本）一致。**一般情形**（无幂律假设）：给定预算 B 选 oracle 子集最大化多数票准确率是**budgeted maximum submodular coverage**——f(S) 单调、次模，贪心（Sviridenko 2004 的 partial enumeration 变体）有 (1−e^{−1}) ≈ 0.632 近似保证；对预算 double-and-guess 可求最小可行预算。
3. **核心机制**：三层——(i) 集成去噪的指数集中（独立性是全部杠杆）；(ii) 幂律假设把"选谁"解析化（一阶条件给 c*，与 ε 解耦：**买哪一档 oracle 是常数决策，买多少随精度对数增长**）；(iii) 无假设时退到次模优化的通用近似。
4. **思想结合点**：两条。第一，**精度阶梯 ↔ oracle 档位的 curriculum**：C3 择时学习早期 policy 远离收敛、信用的方差容忍度高，后期判定门要求 clustered CI 下界 + worst-seed（主线方案 §4.3）——按 AdaProAL 的逻辑，证据预算应前轻后重（早期粗结算探索、formal 段高精度重复结算）。但 volvence 的 prereg 纪律（§0 不变量 3：判定门/seeds/预算看结果前冻结）禁止跑中调整，因此该 curriculum 必须**作为静态阶梯写进 prereg**（预注册的分阶段预算表），不能实现为在线自适应——这是理论与纪律的显式折衷，值得写进 06 综合文档。第二，**独立性假设的反面教训**：Assumption 1 对双专家标注几乎必然不成立（共享文化偏差、同一 rubric 训练），所以 C2 用 Cohen's κ 一致性门（`steering-human-anchor.md` §7：exact agreement ≥ 0.75 且 κ ≥ 0.60，κ 不可识别即门失败）而不是把双专家当独立投票者做 majority vote——本篇的集中界在相关 rater 下失效，恰好论证了 κ 门的必要性：κ 检验的就是"超出偶然的一致"，是对独立性假设的替代品而非近似。
5. **算法结合点**：近期唯一可转写的是**次模覆盖视角下的 C2 unit 选择**：48 units 应覆盖哪些 decision（按 regime、PE 幅度分层、episode 位置等维度），本质是预算约束下的覆盖问题；把"验证信息覆盖度"写成单调次模函数后，贪心选 unit 有 (1−e^{−1}) 保证——落点是 C2 packet 构建前的 capture 采样策略（纯离线、不触学习环路，合法）。**不可直接转写**：加权多数投票需要 ε_i 已知或可估——估计专家 reliability 本身要消耗带真标签的标注（不存在）；oracle 档位选择（c*）需要幂律参数 β,γ——volvence 只有两档 oracle（免费 PE / 专家），拟合幂律无从谈起，c* 公式只做定性参照（"最优档位与目标精度解耦"提示升级 prereg 不必随精度换专家池）。假设核对：persistent（专家满足）、arbitrary noise（比 #30 更贴近现实）、独立性（不满足，见上）。
6. **成熟度与档位**：**B**——curriculum 结构与次模覆盖选样是思想级参照；κ 门的反面论证有引用价值；闭式 c*/M* 因假设不满足只做定性。
7. **风险与不适配**：Assumption 1（独立错误）是全篇杠杆且在人类标注域系统性失效；Assumption 3 幂律在只有两档 oracle 时不可辨识；技术报告未经同行评审；A² 作为基算法的 θ² 依赖已被后续工作（含 #09）改进，成本公式的学习侧因子过时——引用时应替换为 #09 的现代界。

---

### 02. Active Learning with Identifiable Mixture Models（Annals of Statistics 投稿中，2023）

**文件**：**无公开 PDF**（Hanneke 主页标 in preparation；本节仅依据论文集贡献声明分析，结论保守）

1. **基本信息**：Liu Yang, Steve Hanneke, Vittorio Castelli（**杨柳一作**，打破字母序）；Annals of Statistics 投稿中（2023）。贡献声明："active learning under a parametric mixture model assumption ... general upper bound on the risk of an active learning algorithm for identifiable mixture models satisfying regularity conditions ... special case of the mixtures of exponential families."
2. **问题设定与核心结果**（据声明重构，保守）：数据由可辨识参数混合模型生成（分量 = 某种潜在"类型/成分"），学习者主动请求标签；在可辨识性 + 正则条件下给出主动学习风险的一般上界，指数族混合为特例。可辨识混合的意义：无标注数据本身携带关于分量结构的信息（混合参数可从边际分布恢复），主动标签预算只需解决"分量 ↔ 标签"的对应与边界细化——这是"生成假设大幅压低标签复杂度"的经典路线（对照：#09 的分布无关 minimax 不允许任何此类先验）。作为 2023 年投稿的近作，它标志杨柳主动学习工作从分布无关 minimax 转向强结构假设下的统计精细化。
3. **核心机制**：无公开文本，不做推测性重构。
4. **思想结合点**：与 Readable/Learnable 轴的潜在接口是"潜在混合结构 + 少量标签"模式：volvence 的 regime（R14 持久体制身份）与 `steering_condition_belief` 的 subgoal belief 都是"从无标注残差流中可估的潜结构"，若其分布族真的近似可辨识混合，则理论上主动标注（无论 PE 结算还是验证锚）只需按分量后验不确定性分配——与跨簇引用 #34（Bayesian Active Distance Metric Learning，UAI 2007：选相对距离不确定性最大的样本对标注）的策略同族。但在拿到论文全文与 regularity 条件之前，这只能登记为**远期理论准备**，不进入任何 prereg 引用链。
5. **算法结合点**：不可转写（无公开版本；假设匹配度无从核对——"embedding 空间的 regime 混合是否可辨识、是否指数族"完全未验证）。唯一的保守动作：在 06 综合文档的远期清单登记"若 C3 后继工作需要按 regime 分层分配结算预算，先查本篇是否已发表并核对其 regularity 条件对 frozen embedding 空间是否可检验"。
6. **成熟度与档位**：**C**（背景登记）——无公开版，分析基于贡献声明，结论保守。
7. **风险与不适配**：主要风险即"不可得"本身：结论、常数、条件全部未知；参数生成假设与 volvence 的冻结基底表示（非生成式、非参数）距离较远；投稿中意味着结果可能修改。

---

## 簇级小结

### 一、对 Learnable 轴（Gate 4 / 工作流 C）的净贡献清单（按可转化性排序）

1. **查询/标注预算按"决策符号不确定区"分配，且省标签主张只准挂 0-1 型决策目标**（#07 Algorithm 1 + Thm 8/9；#25 Thm 1/2 对偶）。近期落点：C2 packet 选样按 deadzone（±0.02）边界距离分层加密；N+1 PE heldout 结算对"方向已定"episode 降频。判词落点：Gate 4 后继 prereg 的主判据锁定择时方向正确性，禁止滑向信用值拟合误差。
2. **省标签幅度的先验上限模板**（#09 Thm 3/5/7 + star number 等价定理）。高噪声 regime（对话域现状：A1 v1 全 CI 跨 0、v2 d=0.592）改善只有多项式因子（≈1/ν 或 ε^{−α}），与假设类结构无关；指数级改善（O(log(1/ε))）仅存在于 s<∞ + realizable/bounded noise。落点：Gate 4 prereg 的 power 分析用 `ν²/ε²·d` 主项核算结算预算；效应量承诺按高噪声 regime 措辞。
3. **批量-轮次权衡的闭式参照**（#13 Thm 3.1/4.2/5.1）。k 批标签复杂度 `O(k·ε^{−1/k}·θ^{1−1/k}·(d log(1/ε)+log(1/δ)))`，k=1→2 的边际收益最大；次线性成本下主复杂度因子整体进入 c(·)。落点：C2 的 48→(120–240) 两阶段扩量论证；MPS 结算的几何倍增批调度。
4. **"C1 免费主信用 + C2 昂贵验证锚"架构的 oracle 选择理论化，与升级 prereg 骨架**（#30 Thm 1/2；#31）。最优 oracle 由 `c_j/(1/2−α_j)²` 决定：免费 PE oracle 只在其方向噪声率逼近 1/2（= R-C1 风险成真 / C2 方向一致率过低）时才让位——与主线方案 §4.2 升级条件精确同构。远期"专家标注升级为信用源"的单独 prereg 可直接以该判据 + C2 一致率对 (1/2−α) 的估计为骨架。
5. **prereg 纪律的学习论根据**（#24 Thm 1；#16 Thm 1）。学习与自验证的标签复杂度有本质差距，先验/结构知识只补验证一半——S3-E + 冻结判定门正是 volvence 的"结构先验"，C2 之所以 48 units 就有意义，是因为它只验证一维方向假设；熵下界（QueryComplexity ≥ H(P(·)) − O(d)）把"C2 信息量学不动 policy"升格为信息论必然，为验证锚定位提供最强论证。
6. **信用分布保真原则**（#17 Subroutine 2 的 re-noising 教训）。给下游学习器"选择性清洗"的信号可能比忠实噪声信号更糟——为 C1 的 arm-independent 全量入账设计提供理论辩护，并预置对未来"信用过滤"提案的反对论据；tournament 候选选择模板仅当裁决标签取自 N+1 PE 免费面时合法。

### 二、负结果给出的诚实边界（对外表述禁语清单）

- **不能讲"主动学习一定省标签"**：#09 realizable 下 s=∞ 的类（intervals、R^k 线性分类器、axis-aligned rectangles——包括任何嵌入了 interval 结构的策略族）minimax 与被动同阶 Ω(1/ε)；改善与否是（噪声模型 × 假设类结构 × 目标形式）的三元函数。
- **不能讲"对连续信用/PE 改善量的估计可以靠主动调度省预算"**：#25 在 bounded noise 下已判凸目标 proper AL 的 minimax 率 = 被动率 c/n；估计噪声幅度是主动学习本质上帮不上的任务。
- **不能讲"高噪声下省标签是指数级"**：#09 高噪声 regime（TN α ≤ 1/2 / BE ν ≥ √ε）改善只有 `1/(aε^α)` 或 `1/ν` 因子，绝对结算量 `ν²/ε²·d` 依旧庞大；对话域证据（A1/v2 的分辨力现状）正处此 regime。
- **不能讲"被动基线可被免费主动化"**：#17 的 universal activizer 仅在 uniform classification noise 下证明存在，且 Hanneke 2012 已证存在噪声模型使其不存在；对话域噪声结构未知，不得引用该定理承诺改善。
- **不能讲"小样本专家标注可以顺便当学习信号"**：#16 的熵下界表明 96 assignments 的信息量与策略空间熵相差数量级——C2 的验证锚定位不仅是契约选择（R12），也是信息论必然。
- **不能把 #24/#16 的 o(1/ε)/熵刻画直接引为保证**：两者都要求 known prior（+#16 的任意查询、doubling dimension 有限），companion 域只有结构先验的弱类比。
- **i.i.d. 总注**：本簇全部定理假设 i.i.d.（或可交换）标签源；对话 SHADOW 流非平稳，一切引用只做量级/结构参照，非平稳修正归簇 3（#06/#11/#12 漂移系列）。

### 三、供 06 综合文档使用的转化候选表

| 论文 | 轴 | 对象 | 一句话方案 |
|---|---|---|---|
| #09 | Learnable | Gate 4 后继 prereg 的 power 分析 | 按 Thm 5/7 的 `ν²/ε²·d` 主项 + 高/低噪声 regime 二分，预注册"预期省多少结算"的效应量上限与预算 |
| #07 | Learnable | C2 packet 选样 + N+1 PE heldout 结算调度 | 预算集中于 C1 方向 deadzone（±0.02）边界附近的 unit/episode；方向已定区域不复标、降频结算 |
| #25 | Learnable | Gate 4 判词措辞约束 | 主判据只准是 0-1 型择时决策质量；以本篇为据打回一切"信用值估计省标签"表述 |
| #13 | Learnable | C2 扩量 prereg 批结构 + MPS 结算批调度 | 用 `k·ε^{−1/k}·θ^{1−1/k}` 论证两阶段扩量；结算批大小按版本空间收缩几何倍增 |
| #30 | Learnable | 远期"专家标注升级为信用源"prereg 骨架 | 升级判据写成两 oracle 的 `c_j/(1/2−α_j)²` 显式比较，(1/2−α) 由 C2 方向一致率估计 |
| #31 | Learnable | C2 capture 采样 + prereg 预算阶梯 | unit 覆盖写成次模函数用贪心（1−e^{−1}）选样；证据预算前轻后重但必须静态写进 prereg |
| #24 | Learnable | C3/B3 停止与预算条款 | 以 S3-E + 冻结判定门为"结构先验"的 n_ε 型预算，禁止 data-dependent stopping |
| #16 | Learnable | C2 扩量 prereg 功效核算 | 用待判假设集的熵对 96→240 assignments 做信息量下界核算 |
| #17 | Learnable | C1 信用管线设计评审 | 信用分布保真优先于逐点清洗；candidate tournament 只许 N+1 PE 裁决 |
| #02 | Learnable（远期） | 远期登记 | 无公开版；若按 regime 分层分配结算预算，先核对其发表版 regularity 条件 |

### 四、跨簇引用

- **#34 Bayesian Active Distance Metric Learning（UAI 2007，主评在簇 5）**：其"选相对距离不确定性最大的未标注样本对"策略与本簇的标注预算分配同源——都是把查询投向后验最不确定处；在 C2 unit 选样（#07/#31 的转写落点）里它是最朴素的可用 baseline：按 C1 方向后验（deadzone 距离）的不确定度排序采样。
- 非平稳/漂移下本簇结论的失效模式与修正 → 簇 3（#06 nonstationary mixing、#11 drifting distribution、#12 drifting target concept）。
- 择时查询协议（mistakes-vs-queries 权衡）与 gate 的在线择时学习 → 簇 2（#04 online selective sampling、#03 reliable active apprenticeship）。
