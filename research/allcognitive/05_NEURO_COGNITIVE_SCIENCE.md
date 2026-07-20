# 第五卷：脑科学、预测误差、睡眠与群体认知

日期：2026-07-20

## 0. 本卷范围、证据纪律与总裁决

本卷主归属 15 篇：核心 sweep 的 **Active Inference as the Test-Time Scaling Law for
Physical AI Agents**（`2606.22813`），以及
`research/frontier-map-2024-2026.md` Axis E 的 14 篇。除三篇下载受限论文外，分析均以本地
PDF 正文为准。三篇受限论文——BARR、DisRNN、domain-general dopamine——使用前序 frontier
map 已核验的官方全文页、出版社元数据或开放全文镜像，逐篇显式标注 **link-only**；不把摘要之外
未核实的数字补成事实。

本卷坚持三种证据不混写：

1. **神经因果证据**：细胞型特异光遗传、药理、闭环电刺激等受控干预改变神经活动与行为；结论只到
   被操控回路、物种、任务和时间窗。
2. **行为模型 / 表征证据**：行为模型比较、神经解码、几何或 replay–行为相关；可提出机制候选，
   不能证明被拟合的 latent variable 就是脑内真实模块，更不能证明软件架构同构。
3. **理论类比 / 工程证据**：FEP、Active Inference、meta-RL 等给出形式化目标或可运行算法；
   除非有独立生物干预，不得写成脑机制，也不得从“可解释生命系统”跳到“应成为产品总目标”。

总裁决如下：

- **R-PE 得到最强支持，但必须改写成“局部、多内容、多通道的 mismatch 家族”**。V1 sensory PE
  是特征选择性放大；纹状体 dopamine 可携带 reward、punishment 或 value-neutral sensory error；
  它们不是一个可全局广播、可直接最大化的单标量。
- **R1/R5 得到“离线有窗口、相位和抑制门”的生物约束**。睡眠巩固不是无限 replay：同步时序促进
  巩固，BARR 抑制又阻止过同步。写入、重放、抑制/遗忘必须成套。
- **R3 得到行为与表征层支持，而非控制器因果证明**。海马可形成潜在上下文与时间结构的解耦几何，
  人会依环境可预测性调节 temporal abstraction；但几何可解码不等于几何就是 controller。
- **R8 得到局部回路和群体行为的结构类比**。局部 owner、外部痕迹、稀疏社会耦合可以产生全局
  有目的行为；这不证明生物系统使用软件 snapshot，也不许可模块共享可变内部状态。
- **Active Inference/FEP 只保留为可证伪模型族与 research motivation**。培养神经网络给出受限
  预测效度；自动驾驶论文给出离散仿真工程结果。二者都不支持把 free-energy minimization、survival
  或 surprise 设为 VZ 唯一目标，也不支持在线改 frozen substrate。

---

## 1. Cooperative thalamocortical circuit mechanism for sensory prediction errors

**来源：** Furutachi et al., *Nature* 2024，DOI
`10.1038/s41586-024-07851-w`；本地 PDF：
`research/papers/frontier-map/thalamocortical-sensory-pe-nature2024.pdf`。

### 1. 论文事实

UCL Sainsbury Wellcome Centre 团队。头固定、食物限制小鼠在虚拟走廊中学习固定光栅序列，训练
5 天（每日约 `90 ± 48` 次穿越），第 6 天在 10% trial 中把第四位置的预期光栅替换为新光栅，
随后改为 100% 出现以区分 unexpected 与熟悉后的 expected。使用 V1 L2/3 双光子钙成像、
pulvinar 轴突成像及 VIP/SOM 细胞型特异光遗传；不同实验常为 4–9 只小鼠、数百细胞。

### 2. 核心问题

感觉 PE 究竟是无内容的 surprise / “预测减输入”差值，还是对未预测感觉内容的选择性增益？其局部
回路是否需要 pulvinar 输入与皮层去抑制协作？

### 3. 机制拆解

状态是位置条件下的预期光栅与实际视觉输入；更新信号不是一个抽象全局差值，而是 unexpected
刺激到来时，对最选择该刺激的 V1 神经元响应进行放大。pulvinar 提供伴随强 feed-forward
抑制的兴奋驱动；VIP 抑制一组 SOM interneuron，解除对锥体细胞树突的门控。两路同时出现才产生
刺激选择性放大。时间尺度为百毫秒级 sensory readout；论文未证明长期模型更新规则。

### 4. 关键证据

unexpected C 相对熟悉后的 expected C 显著增强 V1 响应（hierarchical bootstrap，
`P < 10^-4`），且不能由跑速、减速、瞳孔或一般唤醒解释；A/B 响应稳定。VIP 或 pulvinar
轴突失活均削弱 PE 响应；VIP silencing 效果与单元 PE 强度强负相关
（`r = -0.78, P = 9.3×10^-120, n = 569`）。单独激活 pulvinar 多为抑制，单独激活 VIP
作用弱；联合激活显著放大高度 stimulus-selective 单元，支持协同而非相加。证据是**局部回路因果**，
但“预测”的形成本身主要由训练设计和响应差异操作化。

### 5. 确证价值

强支持 **R-PE**：mismatch 是一级信号，但其内容保留在局部感觉表征中；支持 **R8** 的 owner-local
处理类比——V1 回路自己产生并调节自身 PE，不要求下游重建一个全局误差。它反对把 curiosity、
reward、salience 与 sensory mismatch 合成同一标量。

### 6. 反证价值

反证“PE 就是无内容 surprise 广播”和“数值相减后统一送往全脑”。也反证仅靠一条 excitatory
error pathway 的简单实现：pulvinar 单独激活甚至抑制网络，必须与细胞型特异去抑制配合。

### 7. 局部可借算法

进入 R-PE spec / benchmark：PE snapshot 应保留 `owner/domain/content/precision/timescale`
而非只有 magnitude；在 shadow 中比较“全局标量 PE”与“局部 feature-selective residual”。
可借协同门控结构，不借具体 VIP/SOM 命名做软件模块仿生。

### 8. 不可外推边界

小鼠 V1、虚拟走廊和视觉光栅不等于自然多模态认知；钙信号不是瞬时 spike 的完整代理；光遗传证明
被操控元件对该 readout 必要/足够到一定程度，不证明 conscious surprise、attention 或一般智能。
更不证明跨库通信应模拟解剖连接。

### 9. 成熟度与裁决

**A（主链神经因果证据）**。行动：把 R-PE 约束为局部、内容保持、precision-gated 的信号族；
禁止实现为唯一全局 novelty/reward scalar。

---

## 2. Augmenting hippocampal–prefrontal neuronal synchrony during sleep enhances memory consolidation in humans

**来源：** Geva-Sagiv et al., *Nature Neuroscience* 2023，DOI
`10.1038/s41593-023-01324-5`；本地 PDF：
`research/papers/frontier-map/sleep-synchrony-memory-consolidation-nn2023.pdf`。

### 1. 论文事实

UCLA/Tel Aviv 团队在 18 名药物难治性癫痫患者中记录 iEEG 和 MTL 单神经元；12 人完成完整
认知测试。每人有 intervention night 与 undisturbed night 的 within-subject 对照。NREM 期间约
90 分钟间歇闭环刺激：以 MTL 慢波 active phase 触发前额叶白质 50 ms、100 Hz 刺激，约每 4 秒
一次；另组接受相同但不锁相的 mixed-phase 刺激。任务是睡前学习 25 对名人—动物关联，次晨测试。

### 2. 核心问题

慢波、spindle 与 hippocampal ripple 的跨区同步只是巩固的相关标志，还是其精确时序对人类陈述记忆
具有因果作用？

### 3. 机制拆解

MTL 慢波提供 cadence；锁相刺激增强全脑 spindle、神经 spike 对慢波的相位锁定，以及 ripple 与
thalamo-cortical oscillation 的耦合。写面发生在离线 NREM；行为 readout 是 recognition 与 pairing。
同样脉冲若相位错误，不等价于有效巩固。

### 4. 关键证据

前额叶白质 sync-stimulation 的 6/6 参与者 recognition accuracy 优于无干预夜
（binomial `P = 0.01`）；mixed-phase 组呈混合或恶化趋势，pairing accuracy 无稳定提升。
sync burst 相对 matched sham 显著增加 spindle-band power（565 contacts，
Wilcoxon `P < 10^-30`）。个体记忆变化与电生理耦合变化高度相关。因有主动闭环刺激与相位对照，
对“同步 cadence 参与 recognition consolidation”是因果证据；样本小且组间并非大型随机试验。

### 5. 确证价值

支持 **R1/R5**：background-slow 不是“空闲时随便跑”的 cron；触发相位、跨区协调和写入窗口会
改变结果。支持将 consolidation 的 cadence、eligible content 与行为结果显式发布，而非只记录
job 是否执行。

### 6. 反证价值

反证“离线计算量越多越好”与“相同 replay/stimulation 可任意平移”。同剂量 mixed-phase 不产生
同效，甚至可能恶化。也反证以单一次晨总分概括所有记忆：recognition 改善而 association 未稳定改善。

### 7. 局部可借算法

进入 background-slow benchmark：同步窗口 vs 随机窗口、相同预算不同 phase 的对照；写入需
记录 cadence、source episode、目标 stratum、前后行为证据。可在 shadow 中加入 phase-aware
consolidation scheduler。

### 8. 不可外推边界

癫痫患者、临床电极位置和小样本限制普适性；刺激的是白质网络，不等于定位唯一通路；recognition
改善不等于长期知识整合、关系记忆或人格连续性。不能从脑节律推出固定的软件 wall-clock 周期。

### 9. 成熟度与裁决

**A（受限但直接的人类神经因果证据）**。行动：R1/R5 的慢循环必须把时序与内容选择当一等控制量，
并保留 no-op / mixed-phase 对照。

---

## 3. A hippocampal circuit mechanism to balance memory reactivation during sleep

**来源：** Karaba et al., *Science* 2024，DOI `10.1126/science.ado5708`。**link-only**：
出版社 PDF 下载受限；本节依据 frontier map 已核验的 Science 官方全文/元数据与开放评论，
不声称逐页本地 PDF 核对。

### 1. 论文事实

小鼠与大鼠 NREM 大规模 hippocampal electrophysiology，覆盖 object displacement、social
memory、T-maze 等任务；官方全文可见的规模包括 `23 sessions/8 mice`、`16 sessions/7 mice`、
`14 sessions/4 mice`。论文识别由 deep CA2 pyramidal cell 与 CCK+（实验中 Sncg+ proxy）
basket cell 组成的新事件 BARR（barrage of action potentials），并进行 event-locked
optogenetic silencing。

### 2. 核心问题

学习后 SWR 增强近期 assembly 的同步重激活；什么机制使其回落，避免巩固变成过同步和网络失稳？

### 3. 机制拆解

SWR 是约 50 ms 的选择性重激活窗口；BARR 最长约 300 ms，与 SWR 反相关交替。CA2 barrage
驱动 CCK+ basket inhibition，优先压制刚在学习/SWR 中升高的 CA1 neuron/assembly，使整夜活动
回到 baseline。机制不是“删除 replay”，而是 replay 与 homeostatic inhibition 的对偶。

### 4. 关键证据

学习相关 CA1 assembly 在 SWR 中重激活、在 BARR 中受抑；其升高随睡眠回落。BARR 时锁定沉默
CCK+ cell 后，回落趋势消失，CA1 synchrony 升高且 memory consolidation 受损。跨三类任务和
两种啮齿类记录增强稳健性；cell-type/event-specific 干预提供回路因果证据，但 Sncg+ 与全部
CCK+ population 并非完全同义。

### 5. 确证价值

强支持 **R1/R5** 的稳态约束：consolidation owner 必须同时拥有增强与抑制/稀释机制；支持
**R-PE** 的 precision control 类比——高激活不是持续提高权重的充分理由。

### 6. 反证价值

直接反证“replay 越多、同步越强、记得越牢”。干预导致的是**过同步且记忆更差**，说明单向强化会
破坏可分性和网络稳定。

### 7. 局部可借算法

进入 R5 spec：每个 slow consolidation batch 同时报告 replay gain、competition/inhibition、
diversity、saturation 与 rollback evidence；benchmark 应包含“增加 replay 反而下降”的压力场景。

### 8. 不可外推边界

啮齿类 hippocampal sleep circuit 不等于软件 memory GC；BARR 不是删除 API，不证明何种语义记忆
应被抑制；不能把 CCK/CA2 名称硬映射成模块。link-only 状态也要求对未在官方页暴露的精确效应量保持
克制。

### 9. 成熟度与裁决

**A（主链回路因果证据，link-only）**。行动：禁止无上限 replay；慢循环必须有独立稳态抑制、
容量退出条件和行为保持测试。

---

## 4. Human hippocampal and entorhinal neurons encode the temporal structure of experience

**来源：** Tacikowski et al., *Nature* 2024，DOI `10.1038/s41586-024-07973-1`；本地 PDF：
`research/papers/frontier-map/hippocampal-temporal-structure-nature2024.pdf`。

### 1. 论文事实

17 名癫痫患者、21 个记录 session，1,456 个 single/multi-unit；主分析含 546 个 hippocampal/
entorhinal unit。六张图被放入 pyramid graph，PRE 随机、6 个 exposure phase 按图 random walk、
POST 再随机；任务与图结构无关。另有 25 名健康隐性知识对照、8 名 explicit benchmark，以及
5 名患者/7 sessions/221 neurons 的 diamond graph 复现。

### 2. 核心问题

人类 MTL 单神经元能否从无明确指令的序列经验中抽取持久、预测性的时间结构？离线 compressed
replay 是否与这种结构形成相伴？

### 3. 机制拆解

初始 image-selective cell 经 exposure 后对 graph-neighbor 增强、对 preferred image 选择性下降；
population geometry 更接近 successor representation 而非纯 geodesic/Euclidean。break 中 30 ms
窗口的三单元顺序重放连接行为秒级顺序与 spike-timing 尺度。海马编码更动态、entorhinal code
更稳定。

### 4. 关键证据

hippocampus/entorhinal 中 relational neuron 高于置换 chance（分别 `n=55, P=0.012`；
`n=42, P=0.024`）。POST direct–indirect decoding 仍显著（`P=8.61×10^-5`）；successor template
在 exposure 多数比较优于 geodesic（多项 `P<0.0001`，10,000 permutations），并与 POST
冲突反应时相关（`rho=.53, P=.0077`）。congruent replay 随学习增加（`P<.001`），incongruent
不变（`P=.195`）。这是强纵向神经/行为相关和结构对照，不是 replay 干预因果。

### 5. 确证价值

支持 **R3**：时间抽象可从经验统计结构中涌现，而非预定义标签；支持 **R5**：短时经验、离线
compressed replay、较持久结构表示可处在不同时间尺度。也支持表征 owner 分工的类比。

### 6. 反证价值

反证“抽象一定是显式规则学习”——参与者不能明确报告 graph；反证简单频次/熟悉度解释，作者用
inner/outer 控制与 diamond graph 距离复现排除主要混淆。也提醒 successor fit 并非唯一机制证明。

### 7. 局部可借算法

为 ETA/R3 设计 hidden-graph benchmark：训练阶段不提供 segment label，测试 latent topology、
future occupancy、POST persistence 与 compressed replay；同时比较 successor/geodesic/frequency
模板，避免只做单一 probe。

### 8. 不可外推边界

患者样本、刺激集很小，unit selection 依赖临床电极；replay detection 是观测性定义，未操控 replay；
template 拟合和行为相关不能证明 successor representation 是唯一算法，更不能证明 latent geometry
本身执行 action selection。

### 9. 成熟度与裁决

**A（人类单神经元主链表征证据）**。行动：R3 benchmark 必须同时验证涌现、预测、持久与行为关联；
不得把可解码 geometry 写成 controller 因果事实。

---

## 5. Abstract representations emerge in human hippocampal neurons during inference

**来源：** Courellis et al., *Nature* 2024，DOI `10.1038/s41586-024-07799-x`；本地 PDF：
`research/papers/frontier-map/hippocampal-abstract-representations-nature2024.pdf`。

### 1. 论文事实

17 名癫痫患者、42 sessions；36 个行为达标 session，2,694 个 well-isolated unit，其中 hippocampus
494，另含 amygdala、preSMA、dACC、vmPFC、VTC。参与者做有随机 context switch 的 reversal
learning；22 sessions 表现出 inference，14 sessions 不表现。部分通过 trial-and-error，部分通过
verbal instruction 学习。

### 2. 核心问题

能够跨 context 推断的高层变量是否在神经 population 中形成可泛化、解耦的表示几何？这种几何与
学习和行为的关系是什么？

### 3. 机制拆解

四个变量为 stimulus、response、predicted outcome、latent context。论文把“抽象”操作化为：
linear decoder 在某变量的未见组合条件上仍能泛化；“disentangled”则是多变量方向近正交。学习后
hippocampal population 同时编码可观察变量与推断出的 latent context。

### 4. 关键证据

只有 hippocampus 在推断形成后同时表现多个变量的 cross-condition generalization；几何质量与
inference behavior 相关，trial-and-error 与 instruction 形成相似几何。多区同录和 inference-present/
absent session 是强对照。证据属于**单神经元 population geometry + 行为关联**；无 hippocampal
cell manipulation，不能称该几何导致推断。

### 5. 确证价值

支持 **R3**：潜在 context 可在快速学习后形成与 observable variable 解耦、支持组合泛化的表示；
支持 R14 的 regime 应是持续内部状态而非 prompt 标签，但支持强度仅到表征层。

### 6. 反证价值

反证“只要 probe 可读出变量就说明抽象”：论文要求跨条件 generalization，而非同分布 decoding。
也反证把语言 instruction 与 trial-and-error 视为必然产生不同表示格式。

### 7. 局部可借算法

将 cross-condition generalization、parallelism/disentanglement、未见组合行为性能加入 R3/R14
评估；latent code 必须在 held-out context combinations 上可读且能预测行为，不能只报告 probe
accuracy。

### 8. 不可外推边界

神经几何是分析坐标，不等于真实模块边界；正交方向不证明 causal independence；癫痫患者和简化
reversal task 不覆盖开放世界主体状态。不能由 hippocampus 独特性推出软件只需一个 central latent
workspace。

### 9. 成熟度与裁决

**A（强人类表征证据，非因果 controller 证据）**。行动：采用其泛化判据；禁止把 probe 几何直接
宣称为 ETA 控制机制。

---

## 6. A recurrent network model of planning explains hippocampal replay and human behavior

**来源：** Jensen, Hennequin & Mattar, *Nature Neuroscience* 2024 / bioRxiv version；
DOI `10.1038/s41593-024-01675-7`；本地 PDF：
`research/papers/frontier-map/recurrent-planning-replay-nn2024.pdf`。

### 1. 论文事实

94 名 Prolific 参与者完成 4×4 动态迷宫；计算模型为 100-GRU meta-RL agent，5 个训练实例，
可花 120 ms 做 policy rollout，物理动作计 400 ms。论文还重分析动态迷宫大鼠 hippocampal
tetrode 数据，每 session 同录 187–333 neurons。

### 2. 核心问题

agent 能否学会“何时值得思考”，而非由实验者固定规划预算？hippocampal forward replay 的内容与
后续行为是否更像 policy-conditioned rollout，而非简单记忆回放？

### 3. 机制拆解

慢时间尺度用 outer-loop RL 学 recurrent dynamics/world-model usage；测试时权重固定，rollout
反馈只改变 hidden state，近似对 hidden-state policy 做 inner policy-gradient update。成功 rollout
提高首个模拟动作概率，失败 rollout 降低；agent 自主权衡 rollout 的机会成本。

### 4. 关键证据

人类离目标更远、trial 首动作时思考更久；agent 在类似位置触发更多 rollout。禁止 rollout 后奖励由
`7.54±0.03` 降至 `6.54±0.11`，随机打乱相同时长 rollout 的时机为 `6.75±0.04`，说明内容与
时机均重要。大鼠 replay 穿墙少于置换、目标过表征（均 `P<.001`）；successful replay 首动作与
后续动作更一致（`P<.001`），未知 away-goal 对照不显著（`P=.129`）。连续第三个 replay 的
goal over-representation 高于首个（`P=.009`）。但 biological replay 未被干预，因果只在人工 agent
强制 rollout 中成立。

### 5. 确证价值

支持 **R1/R3**：慢学习可塑造测试时 fast hidden-state adaptation；抽象/规划触发可由价值与容量
约束学习，而非固定规则。支持“按需要分配 inference compute”，但不是 R-PE 直接证据。

### 6. 反证价值

反证“replay 必须总被后续执行才算 planning”：失败路线也可通过降低其 policy probability 提供
负更新。反证固定 rollout count；时机打乱即损失。也反证把所有 replay 当参数写入：模型主要通过
短时 hidden-state 更新。

### 7. 局部可借算法

在 R3 shadow 中加入 `think` action、显式机会成本和 rollout feedback；评估 successful/unsuccessful
rollout 对 latent policy 的方向性影响、随机时机对照、固定权重下的 fast adaptation。

### 8. 不可外推边界

meta-RL agent 是理论模型，不是已证实 PFC–hippocampus 实现；大鼠 replay–行为证据相关性为主；
grid maze 与显式 world model 远小于开放世界。训练仍依赖 task reward，不能当关系/主体性总目标。

### 9. 成熟度与裁决

**A（强计算—行为桥接；生物机制为 B 强度）**。行动：进入 R3 benchmark/shadow，不直接写成脑
模块事实；保留 rollout cost、失败 rollout 和 timing ablation。

---

## 7. Humans rationally balance detailed and temporally abstract world models

**来源：** Kahn & Daw, bioRxiv 2024，DOI `10.1101/2023.11.28.569070`；本地 PDF：
`research/papers/frontier-map/humans-temporally-abstract-world-models.pdf`。

### 1. 论文事实

104 名 Prolific 招募，100 名进入主结果；约 40 分钟、200 对 traversal/non-traversal trials、22 个
reward block。行为模型比较 full model-based（MB）、successor representation（SR）与 model-free
TD；通过 congruent 与 incongruent block change 操控旧 policy 对未来 occupancy 是否仍可靠。

### 2. 核心问题

人是否固定使用一种 world model，还是依据环境可预测性在逐步 MB simulation 与 temporally
abstract SR 之间资源理性地切换？

### 3. 机制拆解

SR 缓存多步 state occupancy，以灵活性换计算成本；MB 每步递归最大化。non-traversal reward
隔离了必须经 world model 归因的更新。incongruent reversal 改变 island 内最优 boat，使旧 SR
occupancy 失效；短期应降低 abstraction、转向 MB。

### 4. 关键证据

人类同时有 MB interaction（`beta=.743, P<1.32×10^-11`）与 SR interaction
（`beta=.384, P<2.53×10^-5`）。congruent 后 SR 相对权重 `w_SR=.604`，incongruent 后
`.336`，差异 `t(1044)=2.797, P<.006`，93/100 个体同方向；替代 linear-RL 拟合也复现
（`P<.0004`）。这是任务操控对**行为策略权重**的因果证据，但 MB/SR 是模型解释，不是神经模块证明。

### 5. 确证价值

支持 **R3/R1**：时间抽象尺度应由 regime/predictability 调节；稳定环境可用压缩多步表示，结构变化
时回退到更细粒度规划。支持把 abstraction confidence 与 fallback 作为可观测状态。

### 6. 反证价值

反证“更抽象总是更高级”和“一个固定 temporal horizon 最优”。当 future occupancy 失稳时，SR
会产生系统性错误，人类减少而非增加抽象。

### 7. 局部可借算法

建立 controllable predictability benchmark：同预算下改变 transition stability，测 `beta_t/z_t`
是否自动缩短 abstraction horizon；与固定 SR、固定 MB、随机 arbitration 比较。

### 8. 不可外推边界

在线行为、二阶段任务、模型可辨识性限制结论；论文不能区分“两个模块竞争”与单一 linear-RL 连续
参数；未测脑活动，不能把拟合 weight 当神经量。也不能把 reward predictability 外推为关系 regime
的唯一控制变量。

### 9. 成熟度与裁决

**A（强行为模型证据）**。行动：R3 必须支持 regime-dependent abstraction 与细粒度 fallback；
禁止固定抽象层级。

---

## 8. Sub-second fluctuations in extracellular dopamine encode reward and punishment prediction errors in humans

**来源：** Sands et al., *Science Advances* 2023，DOI `10.1126/sciadv.adi4927`；本地 PDF：
`research/papers/frontier-map/human-dopamine-reward-punishment-pe-2023.pdf`。

### 1. 论文事实

3 名 essential tremor DBS 手术患者，在 caudate 以 carbon-fiber voltammetry 每 100 ms 直接测
dopamine；做有 monetary gains/losses 和 reversal 的 probabilistic reward–punishment task。
行为模型另在 42 名健康成人复现。

### 2. 核心问题

人纹状体 dopamine 是否由单一 signed TD-RPE 解释，还是 reward 与 punishment 通过可分离
valence-specific error stream 表达？

### 3. 机制拆解

标准 TDRL 把 gain/loss 合为单轴；VPRL 维护 appetitive 与 aversive 两路 expectation/error，再由
下游组合。reward PE 出现在约 200–300 ms，punishment PE 出现在约 400–600 ms，提示不同时间窗，
而非简单同标量正负号。

### 4. 关键证据

全 trial 的 TD-RPE 正负不分（`P=.24`）；reward trial 可分（ANOVA `P=.016`），punishment trial
不可分（`P=.72`）。VPRL 对 punishment PE 显著（ANOVA `P=.0045`；400–600 ms 各点
`P=.047/.011/.029`）；punishment classifier 仅在 VPRL parsing 下优于 TDRL
（auROC difference permutation `P=.0103`）。行为上 VPRL 优于 TDRL，并在 N=42 复现。
这是直接神经化学测量与模型比较，**没有 dopamine 操控**，故不是通路因果。

### 5. 确证价值

支持 **R-PE** 的 channel separation：reward、punishment/aversive PE 不应先压成单 signed scalar；
时间窗和 valence provenance 应进入 snapshot。也支持 reward 是 PE 的 readout/子域而非 PE 全体。

### 6. 反证价值

反证“一个 Q-value + 一个 TD error 足够解释所有人类 reward learning”。全 trial 聚合甚至会抹去
分离后可见的效应。

### 7. 局部可借算法

为 R-PE 加 valence-partitioned baseline：分别维护 appetitive/aversive expectation、latency、
confidence，再在 credit owner 中组合；benchmark 比较 unified TD 与 partitioned readout。

### 8. 不可外推边界

核心神经样本仅 N=3，且为手术患者、单一 caudate 轨迹；trial 数不能替代独立被试数。模型依赖
task assumptions；不能声称 dopamine 只做 VPRL，也不能将 neurotransmitter 等同软件 event bus。

### 9. 成熟度与裁决

**A-（稀缺直接人类测量，但样本极小、非干预）**。行动：R-PE 默认多通道；结论需在更大样本和
其他区域复现后才升级为强生物约束。

---

## 9. Striatal dopamine signals errors in prediction across different informational domains

**来源：** Costa et al., *Science Advances* 2025，DOI `10.1126/sciadv.adq9684`。
**link-only**：本地 PDF 下载受限；依据 frontier map 已核验的出版社全文/PMC 开放全文与元数据。

### 1. 论文事实

大鼠 sensory preconditioning：先学习无内在价值 cue–cue 关系，再做 reward conditioning 和 probe；
在 nucleus accumbens（NAcc）与 dorsomedial striatum（DMS）记录 dopamine release，并在 probe
阶段操控 lateral orbitofrontal cortex（lOFC）。官方可访问文本确认设计与区域，但本卷不补写未从
本地 PDF 核实的动物数。

### 2. 核心问题

dopamine 是否只编码 value/RPE，还是也编码与当前 reward 无关的 sensory prediction error（SPE），
支持 latent associative learning？

### 3. 机制拆解

在 preconditioning 中，value-neutral cue 被前 cue 预测后 dopamine error 消退；unexpected cue
或交换 predictor 后 error 恢复。后续 conditioning 产生 reward PE；probe 中 NAcc dopamine 还反映
依赖 lOFC 的 inferred value。相同 neuromodulator 可携带不同 informational domain 的 error。

### 4. 关键证据

NAcc 与 DMS dopamine 都与无价值 cue–cue 学习期的 SPE 相关，也与 reward conditioning 的 RPE
相关；predictability 操控使 SPE 消失/恢复。lOFC inactivation 选择性改变 probe 中依赖 inference 的
NAcc dopamine，提供上游回路对该 readout 的因果约束。记录本身是相关，lOFC 操控是局部因果；
不证明所有 dopamine neuron 发同一 domain-general scalar。

### 5. 确证价值

这是 **R-PE** 的关键确证：`prediction mismatch ≠ reward`。同一化学通道可反映 reward-neutral
结构误差，因此 reward/evaluation/curiosity 应由 downstream owner 解释，不能反向定义 PE。

### 6. 反证价值

反证 canonical “dopamine 只等于 RPE”，也反证“只要 dopamine 同时携带两类误差，就能把它们合成
一个标量”：区域、阶段和 lOFC 依赖均不同。

### 7. 局部可借算法

R-PE contract 增加 `domain = sensory|reward|...`、predicted content 与 upstream provenance；
训练用 neutral-transition violation 与 reward violation 正交操控，检查两者是否被错误耦合。

### 8. 不可外推边界

大鼠 conditioning 不等于人类开放世界；dopamine release 是区域混合 readout，不是单细胞统一消息；
lOFC inactivation 只约束 probe inference。link-only 下不报告未核实的 N、效应量和全部统计。

### 9. 成熟度与裁决

**A（跨域 PE 强证据，link-only）**。行动：将 `PE ≠ reward` 写成 R-PE 硬约束；禁止 reward
readout 覆盖 sensory mismatch 原始记录。

---

## 10. Cognitive Model Discovery via Disentangled RNNs

**来源：** Miller, Eckstein, Botvinick & Kurth-Nelson, *NeurIPS* 2023，DOI
`10.1101/2023.06.23.546250`。**link-only**：本地 PDF 下载受限；依据 NeurIPS 官方论文页、
官方 PDF 索引文本与 DeepMind 开源仓库元数据。

### 1. 论文事实

DeepMind 团队提出 Disentangled RNN（DisRNN），以跨时间 excess information penalty、稀疏更新和
少量相对独立 latent variable 约束 RNN。实验包括：由已知 Q-learning/Actor 等认知模型生成的
synthetic bandit data、known bounded-accumulation decision rule，以及大型大鼠 two-armed
bandit choice dataset。代码公开。

### 2. 核心问题

能否从行为数据自动发现稀疏、可命名、可检验的认知 dynamics，同时保持接近 unconstrained RNN 的
预测力？

### 3. 机制拆解

模型在 recurrent state 上惩罚不必要的跨时信息与耦合，使每个 latent unit 稀疏更新、携带较独立的
历史摘要；之后研究者检查 update rule 和 output mapping，将其翻译为候选认知模型。

### 4. 关键证据

在 synthetic known-model data 上恢复生成规则；在 bounded accumulation synthetic task 恢复已知
decision algorithm；在大鼠 bandit 上达到接近最佳人工认知模型与 unconstrained network 的拟合，
并产出可检验 latent dynamics。证据是**系统辨识/模型恢复**，ground-truth recovery 只在 synthetic
成立；真实动物中没有 latent unit 神经操控。

### 5. 确证价值

弱到中等支持 **R3** 的可命名 latent dynamics 与 **R5** 的稀疏历史状态；更重要的是提供方法学：
先用受约束模型提出机制，再独立验证，而非把黑箱 probe 当事实。

### 6. 反证价值

反证“RNN 拟合好就不可解释”，也反证“解释性模型自动等于真实脑模块”。多个 dynamics 可产生相似
choice likelihood；真实数据没有 ground truth。

### 7. 局部可借算法

作为 offline research tool：对 `z_t/beta_t` 轨迹加 sparse-update、independence 与 information
bottleneck，先做 synthetic recovery test，再生成可证伪的 neural/behavioral prediction；不直接进
runtime 决策。

### 8. 不可外推边界

行为可识别性、regularizer bias、简单 bandit domain 限制结论；latent disentanglement 不证明脑区
分离，也不证明 causal owner。link-only 下不虚构 rat 数和具体 effect size。

### 9. 成熟度与裁决

**B（强模型发现工具 / 机制假说生成器）**。行动：进入 rare-heavy/offline analysis；任何发现必须
经 held-out behavior 或神经干预复验后才能进入 spec。

---

## 11. Experimental validation of the free-energy principle with in vitro neural networks

**来源：** Isomura et al., *Nature Communications* 2023，DOI
`10.1038/s41467-023-40141-z`；本地 PDF：
`research/papers/frontier-map/free-energy-in-vitro-neural-networks-nc2023.pdf`。

### 1. 论文事实

RIKEN、东京大学、UCL 团队。rat embryonic cortical culture 置于 MEA；两个独立 binary hidden
source 经 32 个电极按 75/25 mixing 输入，培养网络做 blind source separation。控制组 30 个独立
实验（965 electrodes）；另有 APV、diazepam（7 experiments/127 electrodes）与 bicuculline
（6/129）等药理条件，并操控 source mixing 0/25/50%。

### 2. 核心问题

FEP 能否在预先限定的 canonical network/POMDP 映射下，对未用于拟合的后续 neuronal response 与
effective synaptic plasticity 给出定量预测，而不只是事后重描述？

### 3. 机制拆解

neural activity 对应 hidden-state posterior，effective synaptic strength 对应 generative parameter
posterior，firing threshold 对应 prior。由前 10 sessions 反推 cost/generative model，再沿 VFE
gradient 预测 sessions 11–100；药理改变 excitability，被解释为 prior 改变。

### 4. 关键证据

empirical synaptic trajectory 沿推导出的 free-energy landscape 下降；前 10 sessions 预测末期
connectivity error 小于 4%，预测 session 100 超过 80% neuronal response。药理上/下调 excitability
按模型预期破坏 source inference；改变 mixing matrix 后轨迹仍下降。药理操控提供“excitability 与
任务表现”的因果证据；FEP 身份依赖作者选定的 canonical mapping，属于受限 predictive validation，
不是 FEP 对任意生命系统的证明。

### 5. 确证价值

支持 **R-PE** 作为局部 inference/learning 原始量的理论可行性；支持 **R1** 中 activity 与 plasticity
可处不同时间尺度，并受共同但明确定义的 objective 约束。最有价值的是 out-of-fit-window prediction，
而非“所有系统最小化自由能”的口号。

### 6. 反证价值

反证“FEP 完全不可证伪，所以没有实验内容”——在固定 process theory 后可作定量预测。与此同时也
反证把该结果夸大为普遍生命/意识理论：系统无 action、无 organism-level survival、无复杂行为。

### 7. 局部可借算法

仅进入 research motivation 与 falsification protocol：先固定 generative model、mapping 和未来
预测，再看 held-out trajectory；比较 VFE 与非 FEP cost 的 predictive accuracy。不得直接进
runtime 总目标。

### 8. 不可外推边界

培养细胞、feed-forward rate approximation、无闭环行动；reverse engineering 可能利用 complete
class 式等价，模型选择空间影响“唯一”解释。不能外推 sentience、agency、产品 reward，不能由
局部 Hebbian/homeostatic rule 推出在线更新整个 foundation model。

### 9. 成熟度与裁决

**B+（受限系统中的强预测效度）**。行动：保留为可证伪建模范式；明确禁止把 FEP 设为 VZ 唯一
优化目标。

---

## 12. Hybrid neural–cognitive models reveal how memory shapes human reward learning

**来源：** Eckstein et al., *Nature Human Behaviour* 2026，DOI
`10.1038/s41562-025-02324-0`；本地 PDF：
`research/papers/frontier-map/hybrid-memory-reward-learning-nhb2025.pdf`。

### 1. 论文事实

DeepMind/Oxford/Princeton/UCL 团队。在线收集 880 人，862 通过纳入；4,134 blocks、617,871 valid
trials。四臂 non-stationary bandit，每 block 150 trials。系统比较 handcrafted RL、RL-ANN、
Context-ANN、Memory-ANN、Vanilla RNN，并在 413 held-out blocks 测预测准确率。

### 2. 核心问题

人类 reward learning 是否可由少数 incrementally updated scalar Q-value 概括，还是选择变量与
多时间尺度历史记忆必须分离？

### 3. 机制拆解

Memory-ANN 将 reward-history state `s(r)`、action-history state `s(a)` 与 choice logits `Q/c`
分开；memory state 可携带不同时间尺度的历史并调制 reward-to-value mapping。拟合出的更新并非
标准 `delta = r-Q`：当前 reward 映射到 value，历史 state 调制 gain 与探索—利用变化。

### 4. 关键证据

Memory-ANN held-out accuracy `68.3%`，显著优于 Context-ANN `65.4%`
（`t412=17.9, P<.001, d=.95`），与 Vanilla RNN 不可区分
（`t412=.32, P=.75`）。它复现人类长 action run、cycle 与 sequence compressibility；例如人类与
Memory-ANN compressibility `1.73 vs 1.74, P=.49`。大样本和 held-out 预测强，但模型成分的“记忆”
仍是行为系统辨识；latent injection 是对模型的因果 probe，不是对人脑的操控。

### 5. 确证价值

强支持 **R5**：reward history 不能只存为单 value/RPE；内容、动作与选择 readout 可有独立 memory
variable 和多时间尺度。支持 **R-PE**：即便 PE 有用，也不足以独自概括行为历史。

### 6. 反证价值

反证经典 delta-rule 的充分性，也反证“更大黑箱是唯一办法”：带明确分工的 hybrid model 达到
Vanilla RNN 同等预测。它同时警告不要将 reward memory 与 policy value 合并为同一 owner。

### 7. 局部可借算法

进入 R5 benchmark/spec：分离 episodic event、reward/action history state 与 choice readout；
比较 scalar Q、context-only 与 multi-timescale memory；用 held-out behavior 和 intervention-on-model
验证每个 memory axis。

### 8. 不可外推边界

四臂 bandit、短 session 和 monetary-like point reward 不覆盖关系记忆；拟合 latent 不等于神经
variable；预测准确率提升不证明架构唯一。不能把 evaluation 数据反灌在线学习，也不能据此采用
token-space memory summary。

### 9. 成熟度与裁决

**A（大样本行为反证与强基线）**。行动：R5 禁止以单 Q/RPE 代替历史；memory owner 应发布丰富、
分层且与 choice variable 分离的快照。

---

## 13. Bumblebees socially learn behaviour too complex to innovate alone

**来源：** Bridges et al., *Nature* 2024，DOI `10.1038/s41586-024-07126-4`；本地 PDF：
`research/papers/frontier-map/bumblebee-social-learning-nature2024.pdf`。

### 1. 论文事实

Bombus terrestris 两步 puzzle box：先推无即时奖励的 blue tab，再推 red tab 获蔗糖。三 colony
无示范控制暴露 12/12/24 天（闭箱累计 36/36/72 小时）无个体完成。15 个 demonstrator–observer
dyad 共同训练 30–40 sessions、最多 13.3 小时；5/15 observer 通过无奖励测试并在 solo session
重复行为。

### 2. 核心问题

无脊椎动物能否通过社会学习获得个体在其生命周期实验窗口内无法自行创新的多步程序？

### 3. 机制拆解

第一动作与最终 reward 时间/空间分离，个体 trial-and-error 很难给第一步信用。demonstrator 本身
需 temporary reward shaping 学会第一步，再撤掉 shaping；observer 只看完整行为，未在第一步获
reward。社会传播保存程序结构，可能绕过个体信用发现瓶颈。

### 4. 关键证据

无示范 colony 长时暴露零完整开箱，仅一次 blue-tab move 且未重复；有示范时 5/15 学会。成功者均
来自 squeezing demonstrator（5/10），staggered-pushing 0/5；demonstration 数差异未显著
（`P=.065`），但技术与成功混杂、样本小。证据是强行为对照，不是神经机制；“不能创新”严格说是
在给定实验时长与动机条件下未观察到。

### 5. 确证价值

支持 **R5/R8** 的弱生物类比：程序性知识可经局部社会观察跨个体传递，不需要共享内部状态；某些
skill 的价值在完整序列层显现，不能按单步即时 reward 归因。

### 6. 反证价值

反证“每个复杂行为都必须由个体独立重发现”；也反证纯逐步 immediate reward 足够——连
demonstrator 都需 temporary shaping 才跨过未奖励第一步。

### 7. 局部可借算法

为 procedural memory / curriculum 建 benchmark：最终 reward 稀疏、首步无 reward，比较
independent exploration、demonstration、shaping 后撤除；记录 skill provenance 和完整序列
success，不把每一步都伪造即时 reward。

### 8. 不可外推边界

5 个 learner、单 puzzle、实验室社会观察；不能证明人类式 cumulative culture、teaching、语言模仿
或 colony central controller。社会学习机制未神经操控，也不能推出 agent 间应共享 hidden state。

### 9. 成熟度与裁决

**B+（强行为现象、机制有限）**。行动：作为 sparse-credit 与 procedural transmission 的 benchmark
动机；不把蜂群类比写成跨模块数据通道。

---

## 14. Ants engaged in cooperative food transport show anticipatory and nest-oriented clearing of obstacles

**来源：** Fonio, Mersch & Feinerman, *Frontiers in Behavioral Neuroscience* 2025，DOI
`10.3389/fnbeh.2025.1533372`；本地 PDF：
`research/papers/frontier-map/ant-cooperative-clearing-frontiers2025.pdf`。

### 1. 论文事实

以色列 Weizmann 自然 supercolony 的 longhorn crazy ants（Paratrechina longicornis）。25 个空间
实验记录 292 次 bead clearing；32 个 bottleneck 功能实验比较大 load/crumb 与有/无 bead；
context 实验 4 个 large-load（259 beads）与 6 个 crumbs（12 beads）；个体追踪覆盖 155 次
ant–bead decision、167 个 clearer 等。

### 2. 核心问题

面向巢的预先清障是否要求单蚁知道大 load 和未来路线，还是可由局部 pheromone cue 与 task
allocation 涌现为 colony-level goal-directed behavior？

### 3. 机制拆解

大 load 招募提高 fresh pheromone marking；新蚁在 bead 附近遇到刚落下的 mark 后转为 clearer。
首次触发后约四分之一成为 serial clearer，可在无新 mark 下连续工作。pheromone 是环境中的短寿命
外部表示，连接 load–nest 可能路线；单蚁无需持有最终目标。

### 4. 关键证据

有 bead 时大 load bottleneck 中位通行时间 `182.3 s`，空通道 `12.8 s`；crumb 为 `2.0 vs 1.0 s`。
large-load 清除数是 crumbs 的 32 倍（`P<.00003`），距离 >6 倍（`P<10^-8`）。155 次决策中，
97.2% 的 clearing 前 2 秒/20 mm 内有 fresh mark，Fisher `P<10^-36`；ant–ant contact 不预测
clear (`P>.77`)。仅 45% clearer 曾触碰 load；无 load 的 tuna-oil 高招募条件也诱发清障。
这些是多重行为操控与时序证据，但没有直接化学阻断/人工施加已鉴定 pheromone，故“pheromone
分子因果”仍弱于神经光遗传。

### 5. 确证价值

支持 **R8** 的结构约束：全局目的可由 owner-local 状态、环境痕迹和稀疏耦合产生，不要求共享全局
内部状态；支持 **R1/R5**：fresh mark 是快、易衰减的 external memory，serial clearer 是更持久的
task state。

### 6. 反证价值

反证“群体有目的行为必有 central planner”与“每个执行者都要知道最终目标”。也反证把 ant density
本身当触发信号：接触率不区分 clear/no-clear，fresh mark 才高度关联。

### 7. 局部可借算法

进入数字蚂蚁 benchmark：只允许局部 immutable observation、短 TTL trail snapshot 与 owner-local
task state；做 no-load/high-signal、load/no-direct-contact、density-matched 对照，检验 global
homing 是否涌现。

### 8. 不可外推边界

pheromone 是可被环境读取/改写的物理痕迹，不等于软件模块直接共享 memory；论文未排除所有未测
social cue，相关时序不等于特定分子充分性；单一 supercolony 和物种限制泛化。不能称 colony
“拥有与人同类的内部 world model”。

### 9. 成熟度与裁决

**B+（强群体行为机制证据）**。行动：作为 R8 局部耦合/外部痕迹 benchmark；禁止据此引入共享
mutable global state 或 central ant controller。

---

## 15. Active Inference as the Test-Time Scaling Law for Physical AI Agents

**来源：** Hashash et al., arXiv `2606.22813v1`，2026-06-22；本地 PDF：
`research/papers/sweep-2607/active-inference-test-time-scaling-2606.22813.pdf`。

### 1. 论文事实

Virginia Tech、WPI、Khalifa University/CentraleSupélec、UCL、Monash 等团队，含 Karl Friston。
论文为 53 页 arXiv v1，提出 Active Inference test-time scaling，并在 64-state 离散自动驾驶
intersection 仿真中与 Q-learning、Bayesian RL 比较。三者训练 6,000 episodes；未报告真实机器人、
人类或动物样本。

### 2. 核心问题

物理 agent 遇到训练分布外情形时，能否由 prediction error 触发 world-model reasoning，只在需要时
把 base policy 更新为 posterior policy，并把已解决经验在慢时间尺度巩固？

### 3. 机制拆解

当前 surprise 以 VFE 上界；超过阈值 `epsilon` 后枚举 policy，以 EFE 的 preference risk +
ambiguity/epistemic term 评分，得到 `q(pi) ∝ exp(-gamma G(pi))`，再与 base policy 做 soft
Bayesian update。fast inference 只改 policy belief；slow learning 以 Dirichlet pseudo-count 更新
observation model A、transition model B 与 base policy。论文再用 PFC/BG/dopamine 作生物类比，
但未以神经数据检验。

### 4. 关键证据

仿真 state 为 distance×velocity×light×pedestrian，共 64 states，3 actions；训练中 pedestrian
只在红灯，测试时绿灯 jaywalking。Q-learning reward `-21.8`、success 3%；Bayesian RL `-18.2`、
5%；方法 `22.9`、100%。推理仅覆盖 63.3% timestep，相对始终 inference 的 Bayesian RL 少约
36%。50 次重复后 surprise 由 1.95 降至约 0.2 bits，绿灯 pedestrian prediction 由 .25 到 .9，
slow-down policy 由 .3 到 .92。证据是**单一手工离散仿真**，无随机多环境统计、真实 OOD 或生物
验证；“scaling law”名称强于证据覆盖。

### 5. 确证价值

理论上支持 **R-PE/R1/R3**：PE 可触发快 inference，熟悉后转慢学习；reasoning compute 可按
surprise 分配；fast belief update 与 slow parameter update 可分层。它提供可实现 baseline，而不是
VZ 主链证据。

### 6. 反证价值

论文把 survival/preferred states 作为总目标，并允许 world model 与 policy 在部署中持续写入；
这正是对 **R2/R5/R10** 的压力测试：若 surprise 来自 sensor fault、攻击或不可控噪声，直接巩固会
污染 substrate。其 EFE 把 preference matching 与 epistemic value 放入同一 objective，也可能把
evaluation/reward 反灌 PE。

### 7. 局部可借算法

仅进入 research benchmark：实现 surprise-triggered posterior policy 的 shadow baseline；加上
frozen substrate、owner-local controller、ModificationGate、replay provenance、quarantine 和
rollback 后再比较。必须测试 sensor corruption、adversarial surprise、preference misspecification、
不可预测噪声与 threshold drift。

### 8. 不可外推边界

离散 MDP 已知 state space，真正新 state/model expansion 被明确排除；100% success 来自单一设计
情形。PFC/BG/dopamine 是理论映射，不是神经因果证据。NESS/Markov blanket 推导不证明 AI 应有
“生存欲”，更不证明 free energy 是关系型数字生命的唯一价值函数。禁止在线更新 frozen foundation
model，禁止 token/prompt RL。

### 9. 成熟度与裁决

**B-（有形式推导的工程概念验证兼反例）**。行动：只进 research motivation 与 adversarial
benchmark，不进 runtime；若试验，必须 SHADOW、bounded、可回滚，并由 owner/gate 隔离写面。

---

## 16. 跨论文综合：对 VZ R-ID 的生物学约束

### 16.1 R-PE：PE 是原始信号，但不是一个数

由 thalamocortical PE、human valence-partitioned dopamine 与 domain-general dopamine 可得：

1. **局部性**：PE 由具体 sensory/reward circuit 按本域预测产生；V1 的 PE 保留 stimulus content。
2. **多域性**：value-neutral sensory error 与 reward error 均可存在，甚至共享 dopamine readout，
   但不能因此视为同一目标。
3. **多通道性**：reward/punishment 的符号、时延和学习模型可分；聚合会掩盖结构。
4. **precision/gating**：同一输入是否被放大取决于 pulvinar×VIP/SOM 协作，不是 magnitude 单独决定。
5. **读出隔离**：reward、curiosity、salience、evaluation 是对 PE 的假设性 readout；必须保留原始
   mismatch 与 provenance，禁止消费者反向重写 producer 的 PE。

因此，R-PE snapshot 最少需要：
`owner/domain/predicted_content/observed_content/residual/precision/latency/provenance`。
这是一条软件契约建议，不声称脑内以 dict 或 snapshot 编码。

### 16.2 R1：多时间尺度必须有写面与退出条件

- sensory PE：百毫秒级局部 readout；
- rollout/recurrent state：秒内 fast adaptation，可不改参数；
- task/session memory：跨 trial 的丰富历史与 abstraction arbitration；
- sleep consolidation：离线、相位敏感的中慢写入；
- repeated resolved surprise：只在证据累积后进入慢参数更新。

生物证据不支持“所有时间尺度统一在线梯度”。相反，planning model 显示 fixed weights 下 hidden-state
优化，睡眠研究显示离线窗口，BARR 显示同一窗口内还需反向稳态过程。VZ 的 online-fast /
session-medium / background-slow / rare-heavy 分层应保留独立 owner、预算和退出条件。

### 16.3 R3：时间抽象可涌现，但必须可降级

hippocampal temporal graph 与 abstract geometry 支持从经验中形成 predictive/latent structure；
人类 SR–MB 研究则给出必要反例：当未来 occupancy 不稳定时，降低抽象更合理。R3 不能只追求
“更高层 latent”，而要同时测：

- held-out combination generalization；
- latent topology 与 future occupancy；
- abstract representation 对行为的增益；
- predictability 下降时 horizon/`beta_t` 是否收缩；
- 失败时是否回退细粒度 planning；
- geometry 是否只是 probe correlate，而非被误当 causal controller。

### 16.4 R5：记忆不是 scalar value，也不是无限 replay

Memory-ANN 反证单 Q/RPE 概括历史；temporal-structure 论文说明 replay 可压缩经验结构；sleep
synchrony 说明时序决定写入效应；BARR 说明巩固必须配稳态抑制。因此 R5 至少要分：

- 原始 episode/provenance；
- 多时间尺度 history state；
- task/procedural memory；
- replay proposal 与 replay outcome；
- consolidation eligibility/cadence；
- inhibition/forgetting/saturation；
- held-out behavior 证据。

任何“把更多日志喂回模型”都不等同巩固；任何“replay 次数增加”都不能单独作为成功指标。

### 16.5 R8：局部 owner 与外部痕迹可产生全局行为

V1 回路展示 local content-preserving PE；蚂蚁展示 local cue + environmental trace + owner-local
task state 可产生 nest-oriented clearing；蜂类展示程序可社会传递但不需共享 hidden state。这些研究
支持 R8 的**结构原则**：

- producer 自己生成有内容的信号；
- consumer 只接收受限可观察量；
- 全局效果不要求 central controller；
- 环境/共享介质中的痕迹必须有来源与衰减；
- 个体/模块保持内部状态所有权。

但生物学不证明 immutable snapshot 是神经或昆虫的真实实现。snapshot 是 VZ 为可测试性、隔离和
回滚选择的工程契约；不能把“蚂蚁 pheromone”当成共享 mutable global store 的许可。

---

## 17. 禁止外推清单

1. **禁止从神经相关写成神经因果。** decoder、representational geometry、dopamine correlation、
   replay–behavior association 都不是干预。
2. **禁止从局部因果写成全脑统一机制。** V1 VIP/SOM/pulvinar 结论只覆盖该感觉回路和任务。
3. **禁止把 PE、reward、surprise、salience、curiosity、evaluation 互作同义词。**
4. **禁止把 dopamine 写成单一 signed scalar bus。** 区域、valence、domain 和时间窗均有异质性。
5. **禁止把可解码 latent geometry 写成 action controller。** 表征可读不等于控制因果。
6. **禁止把 hippocampal replay 写成“越多越好”。** BARR 直接显示过同步损害巩固。
7. **禁止把睡眠节律机械复制成固定 cron。** 生物约束是 cadence/eligibility 重要，不是某个绝对周期。
8. **禁止把培养神经网络 FEP 结果外推为 sentience、agency 或唯一生命目标。**
9. **禁止把离散自动驾驶 Active Inference 仿真称为普适 scaling law 已获验证。**
10. **禁止把 survival/preferred state 设为 VZ 唯一 reward。** 关系、边界与主体性不能被单目标吞并。
11. **禁止由 active inference 推出部署时可在线更新 frozen substrate。** 慢写入仍需
    ModificationGate、证据、隔离和回滚。
12. **禁止把行为模型 latent variable 当真实脑模块。** MB/SR、DisRNN、Memory-ANN 都是候选解释。
13. **禁止把动物任务表现直接外推人类关系认知。** 蜜蜂 puzzle 与蚂蚁清障不是 trust、consent 或
    theory of mind 证据。
14. **禁止从群体涌现推出 central colony mind。** 全局目标性可在个体无最终目标表征时出现。
15. **禁止把 pheromone 类比成跨模块共享可变状态。** VZ 仍必须通过不可变 snapshot 与 owner contract。
16. **禁止把 trial 数、cell 数当独立生物样本数。** 尤其人 dopamine 核心样本只有 3 人。
17. **禁止在 link-only 论文中补造未核实 N、效应量或统计。** BARR、DisRNN、domain-general
    dopamine 的结论强度受可访问正文范围约束。
18. **禁止让 evaluation 反向成为在线学习源。** 本卷所有行为/神经指标首先是 readout；进入学习前
    必须另有 owner、门控和 provenance。

## 18. 本卷最终裁决

最可靠的生物学结论不是“脑已经替 VZ 选好了算法”，而是五条负约束：

1. mismatch 必须局部、有内容、可追溯，不能塌缩为总 reward；
2. 快状态更新、session memory、离线巩固与慢参数更新不能混成同一写面；
3. temporal abstraction 必须随环境可预测性调节，并保留细粒度 fallback；
4. memory 必须同时拥有 replay 与抑制/遗忘，且以 held-out behavior 而非写入量验收；
5. 全局适应可由局部 owner 和受限通道涌现，不需要也不应创建共享可变“全局大脑状态”。

据此，本卷对 **R-PE、R1、R3、R5、R8** 提供约束与 benchmark 依据；不授权修改这些 R-ID 的
owner 边界，不授权 FEP/Active Inference 进入 runtime 主链，也不授权任何生物类比绕过不可变
snapshot、frozen substrate 或 ModificationGate。
