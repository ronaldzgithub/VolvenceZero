# Symbolica AI — 深度分析

- **分组 / 成熟度 / 一句话主张**：A（脑启发 / 结构化推理）｜成熟度 **低**（仅 1 篇锚定出版物，且是 *position paper*，无规模化实证）｜用范畴论（monad/functor 代数）把"架构 = 满足代数定律的对象 + 类型化态射"形式化，主张以**可证不变量 / 结构约束**替代纯统计缩放。
- **主要创作者 + 血统**：George Morgan（创始人/CEO，非该文作者）；论文作者 Bruno Gavranović、Paul Lessard（Symbolica）+ Andrew Dudzik、Tamara von Glehn、João Araújo、Petar Veličković（Google DeepMind / Cambridge）。血统为 **范畴论 + 几何深度学习（GDL）一脉**。与 VZ 的共振点在 **R8/R11/R15（契约 / 可命名状态 / 可审计可回滚）** 与 **R9/R10（有界、可证自修改）**——把"模块边界 / 自修改约束"形式化为代数定律；对立点在其 **"结构替代规模"** 的隐含赌注，与 VZ R2（冻结大基底）相反。

---

## 1. 核心逻辑（论文级 · PDF-grounded）

### Categorical Deep Learning is an Algebraic Theory of All Architectures（arXiv:2402.15332v2，ICML 2024，PMLR 235）

- **问题**：深度学习架构有两种互不衔接的描述方式——**自上而下**（用模型必须满足的约束来规定，如 GDL 用对称群的等变约束）与**自下而上**（用实现，即 tensor 运算序列 / autodiff 框架）。作者主张：迄今缺乏一个能**同时**桥接"约束规格"与"实现"的统一理论（§1.1–1.2）。
- **方法 / 机制**：提出用范畴论——精确说是 **2-范畴 `Para`（参数化映射）中取值的 monad 的泛代数**——作为统一理论。逐层构造：
  1. **Monad 代数 → 等变性**（§2.1）：群作用 monad `G × −` 的代数即群作用；**代数同态（algebra homomorphism）= 等变映射**，从而完整复现 GDL 的核心目标（不变性 / 等变性），并覆盖 GNN、Spherical CNN、G-CNN（Appendix C）。
  2. **Endofunctor (余)代数 → CS 构造**（§2.2）：把 monad 放宽为任意 endofunctor，其代数/余代数恰好"重新发现"列表（`1 + A × −`）、二叉树（`A + (−)²`）、自动机（Mealy/Moore 机、流）。**代数同态 = fold，余代数同态 = unfold**；选定数据结构即对对应神经网络的**控制流施加结构约束**（fold → 折叠式 RNN，unfold → 递归 / 全循环 RNN）。
  3. **2-范畴 `Para` → 参数与权重共享**（§3）：1-态射是参数化函数 `(P, f: P×A→B)`，2-态射捕获**重参数化 / 权重绑定（weight tying）**（用 copy map `Δ_P` 形式化）。**lax 代数**足以从泛性质"第一性原理"导出循环 / 递归网络的**整体结构**（不止单层）；权重绑定的本质是一个 comonoid 结构（Theorem G.10）。
- **关键结果（PDF 内具体内容）**：注意——**本文是"Position"立场论文，无 benchmark、无训练实验、无数字指标**。其"结果"全部是**形式化的再推导（re-derivation）**：(a) 复现 GDL 的等变/不变约束；(b) 用 (余)代数统一表达列表 / 树 / Mealy / Moore / 流；(c) 在 `Para` 中给出权重绑定/共享的正确性形式判据，并把 RNN/递归网络作为"自由参数化 monad 的 lax 代数"导出（§3.2，Fig.1）。
- **局限（PDF 自陈 + 结构性）**：
  - **零实证**：全篇无任何实验、无新架构跑分；只论证范畴论"足够表达"已知架构，未证明它能**改进**学习 / 泛化 / 缩放。
  - **范畴选择问题**："任何结果都依赖选对范畴，正如 GDL 依赖选对对称群"（§4 首段）——把最难的部分（选哪个代数）外包给了**人工建模选择**，无学习程序。
  - **全是"愿景"**：§4 New Horizons 的强主张（学习"只产出良类型函数 / 可验证逻辑论证 / 代码"的网络、更细粒度公平性、"可验证 AI 而非仅可解释 AI"）均以 "we hypothesise / we hope" 措辞给出，无任何落地证据。
  - 篇幅自陈："space constraints prevent us from adequate level of detail"（§3.2）。

---

## 2. 与 VZ 的关系（三视角）

> 纪律：**先反证**。Symbolica 成熟度低、无规模化证据，必须诚实对待——"可证不变量 / 范畴约束"目前是**一篇没有实证支撑的论点（thesis）**，不是被验证的机制。

### 2.2 反证（红队）— 先行

| # | 反例（PDF-grounded） | 裁决 | 边界条件 |
|---|---|---|---|
| RB-1 | **"可证不变量"目前是论点而非机制**：全文是 position paper，零实验、零跑分，只"再推导"已有架构（GDL/RNN/自动机）。99 综合表把 Symbolica 列为 R9/R10/R15"形式约束"背书与 primitive 7"类型/范畴接口"证据——但本文**提供不了任何经验证据**支撑"形式约束真能让自修改可证有界"。 | **needs-boundary-condition** | VZ 可把范畴/类型接口当作**设计期工程纪律**，但**禁止**据此宣称运行时"可证 / provable"保证；99 中对 Symbolica 的背书须降级为"概念建模工具"，不计入跨领域独立实证。 |
| RB-2 | **"结构替代规模"赌注 vs R2**：Symbolica 的整体叙事（George Morgan 公开主张）是以范畴结构 + 可证不变量**替代纯缩放**。这与 VZ R2 的核心赌注（**冻结的大规模预训练基底**）直接对立——若"结构能替代规模"成立，VZ 的基底前提就可疑。 | **survives** | 本文**未提供任何**支持反缩放论点的证据；而 R2 已被 Group C（ESM/Evo/AlphaFold 等）在非语言模态上跨域实证。结论：结构与规模是**互补**而非替代——范畴约束用于**基底之上的控制器层契约**，不动冻结基底。 |
| RB-3 | **范畴/约束不可学，须手工选**：§4 自陈"依赖选对范畴"，框架把核心难点留给人工建模，无学习程序。这与 VZ 的 ETA 精神（行为模式应**涌现**、禁止硬编码规则——见 `no-keyword-matching-hacks` 规则）以及 R3/R4（控制在 latent `z_t` 空间学习）相冲突：硬编码的范畴约束 ≈ 硬编码的归纳偏置。 | **needs-boundary-condition** | 范畴/类型约束只能作为**设计期的快照 schema 与模块边界纪律**，不能冒充**运行时学到的内部表示**。`z_t/β_t` 仍须从数据涌现；类型契约只约束其"接口形状"，不替代其"内容学习"。 |

### 2.1 确证（先进性背书）

> 均为**概念性 / 形式性**背书（来自纯数学 + CS，非 ML 实证），强度弱于 Group C 的跨模态实证，引用时须标注"aspirational"。

- **R8 / R11（契约优先、可命名可发布状态）**：范畴论被定位为"一套久经考验、学一次即可跨学科可靠复用的接口系统"（§1.3）。"模块 = 满足代数定律的对象 + 类型化态射"正是 VZ 快照契约隔离（`vz-contracts`）的**数学母语**——独立于 ML 社区，从抽象代数侧印证"以代数定律 + 接口约束界定模块边界"的合理性。
- **R3 / R4（结构化、非纯 token 的控制）**：(余)代数视角把神经网络的**控制流由数据结构约束**（fold/unfold over 列表/树/自动机）。RNN/Mealy 机作为余代数 = **基于状态的潜在动力学**，而非 token 串——为"控制发生在结构化 latent 空间"提供形式语言（概念级，非实证）。
- **R9 / R10 / R15（有界、可审计、可回滚自修改）**：代数同态 = **保结构（structure-preserving）映射**；"良类型函数"由域/陪域代数选择**按构造保证**（§4）。这给"自修改必须保持某不变量"提供了一个**形式建模范式**——若能落地，是 ModificationGate"按构造满足前置条件"的理想数学外壳。**注意：本文只给出愿景，未证明可落地。**

### 2.3 局部算法借鉴（算法级解耦）

> 剥离"范畴论替代缩放"叙事，仅取**可作为快照/自修改"可证有界、可审计"工程脚手架**的机制。五元组：机制 → 目标 spec → 落地动作 → 预期收益 → 风险/前提。

1. **快照槽 = 代数对象，跨模块态射 = 代数同态（保结构约束）**
   - **目标 spec**：[`contract-runtime.md`](../../../docs/specs/contract-runtime.md)、[`semantic-state-owners.md`](../../../docs/specs/semantic-state-owners.md)
   - **落地动作**：把每个快照 slot 建模为带代数定律的对象，跨模块 propagate 的 value 变换要求是"代数同态"（保持槽不变量）；schema 违反 = 边界处的**类型错误**，fail loudly。
   - **预期收益**：把契约漂移 / schema 不匹配从"运行时静默回退"提前到**契约期/构造期可检测**，与 `no-swallow-errors` 规则同向加固 R8。
   - **风险 / 前提**：范畴/代数须**人工选定**且无运行时学习（RB-3）；引入形式化开销；仅作 schema 纪律，不可宣称端到端"provable"。

2. **(余)代数 + 同态约束建模有界状态机（regime / 控制器）**
   - **目标 spec**：[`cognitive-regime.md`](../../../docs/specs/cognitive-regime.md)、[`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md)
   - **落地动作**：把 regime/控制器形式化为余代数 `state → output × next-state`，并以同态约束声明"状态转移须保持身份不变量"，作为 R14 持久体制身份的形式化检查点。
   - **预期收益**：为"regime 切换不破坏持久身份"提供可声明、可校验的结构约束（接口层面）。
   - **风险 / 前提**：纯属概念外壳，须经实证验证；`β_t/z_t` 仍须涌现学习，范畴约束只约束接口形状不替代内容。

3. **参数化态射 / 权重绑定的正确性判据 → 自修改前置条件**
   - **目标 spec**：[`credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md)
   - **落地动作**：借 `Para` 中"参数变换何时保持代数（weight-tying 正确性，§3.1 + Theorem G.10）"的判据，作为 ModificationGate 的**前置条件模板**——adapter-delta 在应用前须证明落在某等变/保结构类内方可通过。
   - **预期收益**：给"有界自修改"一个**按构造满足的前置条件**，强化 R9/R10/R15 的"单调改进 + 可回退"。
   - **风险 / 前提**：需把 VZ 控制器形式化为参数化态射，工程量大且当前纯理论；优先级低于已实证的 trust-region / LoRA-without-regret 判据（来自 Thinking Machines）。

---

## 3. 一句话定位

> **Symbolica 是 VZ 契约/自修改轴（R8/R11/R15 + R9/R10）的"数学愿景供应商"而非"证据供应商"：它用范畴论为"模块=代数对象、跨模块=保结构态射、自修改=保不变量变换"提供了优雅的形式母语，但全篇是无实证的 position paper——可借其"类型化接口纪律"加固快照 schema 与 ModificationGate 前置条件，但必须把"可证 / 替代缩放"的主张降级为 aspirational，不得当作已验证机制引用。**

---

## 附：本地论文清单（同目录 PDF）

| 论文 | 年 | ID | 文件 |
|---|---|---|---|
| Categorical Deep Learning is an Algebraic Theory of All Architectures（ICML 2024 position paper） | 2024 | arXiv:2402.15332 | `categorical-deep-learning-algebraic-theory-of-architectures-2402.15332.pdf` |
