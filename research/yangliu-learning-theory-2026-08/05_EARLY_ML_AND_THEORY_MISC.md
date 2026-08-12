# 05 · 早期 ML 与 CS 理论杂项（簇 5，16 篇）

> 总索引：[00_PAPER_INDEX.md](00_PAPER_INDEX.md)。跨簇：簇 1 [01_ACTIVE_LEARNING_LABEL_COMPLEXITY.md](01_ACTIVE_LEARNING_LABEL_COMPLEXITY.md)、簇 2 [02_RL_APPRENTICESHIP_SELECTIVE_SAMPLING.md](02_RL_APPRENTICESHIP_SELECTIVE_SAMPLING.md)。
> 阅读口径：C 档背景登记——每篇只读摘要与引言首段（#28/#33 大文件仅首页）；#37/#40 无公开 PDF，依贡献声明登记。个别篇目在总索引中标 B（思想级参照），不改变簇级判词。
> Volvence 侧对照文档：[appendable-readable-learnable-steerable.md](../../docs/appendable-readable-learnable-steerable.md)（四轴与证据状态）、[semantic-embedding-backend.md](../../docs/specs/semantic-embedding-backend.md)（语义嵌入后端）、[steering-human-anchor.md](../../docs/specs/steering-human-anchor.md)（人类标注只作验证锚）。

## 簇定位

本簇收录杨柳 2004–2018 年间的 16 项工作，横跨华中科技大学本科期的图像配准（#40）、MSU 度量学习/半监督时期（#26 之外的大多数）、CMU 转型期的查询式学习（#22）与 IBM/Yale 时期的组合优化与几何杂题（#19/#20/#26/#27/#29）。整体按 C 档背景登记：没有一篇携带可直接进入 volvence 四轴（Appendable / Readable / Learnable / Steerable）的机制。但簇内藏着两条有单点价值的线：(a) **#34** 的"后验不确定性驱动成对标注选择"，是主线提升方案 C2 人类验证锚"选哪些 turn 送标"预算分配的思想源头之一；(b) **#32** 的"弱相关无标注数据利用"，是远期"非 companion 语料辅助 companion 域"的参照。其余各篇按历史谱系登记，不强行建立关联。

## 谱系：MSU 应用期 → CMU 理论转向

MSU 时期（2006–2010，导师 Rong Jin，长期合作者 Rahul Sukthankar）的作品（#38/#34/#35/#33/#36/#28/#39/#32）全部围绕同一个实践痛点展开：**标注昂贵，成对约束（pairwise constraints）是最自然的监督单位**——度量学习靠 pair 学、码本靠类标签修、半监督靠廉价无标注语料补。她此期已系统梳理 DML 谱系（见补充材料 `papers/supp-dml-comprehensive-survey-msu2006.pdf`，2007 年 overview 短文），且 #34 已经在问"哪些 pair 值得送标"——这正是查询复杂度问题的应用雏形。2009 年转入 CMU（Avrim Blum & Jaime Carbonell）后，这个痛点被形式化为"与 oracle 交互的数学理论"（2013 博士论文题名即此）：标注预算 → 标签复杂度与 proactive learning（簇 1），成对相似性 → 表征特定成对查询（本簇 #22），选择送标时机 → 在线选择性采样（簇 2）。CMU 之后（IBM 2014 起、后 Yale）她把同一套复杂度视角带进分配、匹配、嵌入失真等杂题（#19/#20/#26/#27/#29）。因此本簇不是随机杂集，而是她后期以查询复杂度为中心的理论纲领的经验土壤：先在应用里反复撞到"监督昂贵"这堵墙，再回头为墙本身建理论。

## 逐篇简评

### 19. Online Allocation and Pricing with Economies of Scale（WINE 2015）

**文件**：papers/19-online-allocation-economies-scale-wine2015.pdf

Blum、Mansour、Yang 三人合作，研究边际成本递减（economies of scale）商品的离线/在线分配：顾客 unit-demand、估值 iid 采样，目标是近最小成本地满足所有顾客（"Thrifty Santa"问题），另给出预算约束下社会福利的 bicriteria 近似。方法核心是一个结构性质——边际成本非增时，最优分配可压缩为一个"商品排序"（每个顾客拿序中第一个在其清单上的物品）——在线情形先用一段初始样本学出近优排序、再冻结执行，由排序规则类的低复杂度保证 uniform convergence；当边际成本下降不过快（如第 t 份成本 t^-α，α∈[0,1)）时达到常数近似。与 volvence 的距离：远，机制设计题材无对接面。唯一可记的一句是其"先采样学结构、再冻结执行"的 sample-then-commit 模式与 volvence 的时间尺度分层（rare-heavy 离线训练 artifact + 在线冻结执行）是范式上的同构，但这只是学习理论的标准论证形态，不构成可借鉴机制。

### 20. Risk-Averse Matchings over Uncertain Graph Databases（ECML PKDD 2018）

**文件**：papers/20-risk-averse-matchings-ecml2018.pdf

Tsourakakis、Sekar、Lam、Yang（时在 Yale）提出在不确定加权（超）图上找**期望奖励最大且风险有界**的匹配，模型统一覆盖离散与连续分布（可容纳向边权注入噪声的隐私应用），推广了以往仅 Bernoulli 边的 possible-world 模型。问题 NP-hard；给出图上 1/3 近似、近最优运行时间的 1/5 近似，以及 rank-k 超图上近线性时间的 Ω(1/k) 近似，并以合成实验刻画 reward-risk 权衡。动机例子说得很清楚：期望奖励最高的匹配可能以 1/4 概率颗粒无收，稳健解宁可让渡期望值换确定性。与 volvence 的距离：远亲，如实写一句——"最大化期望收益 subject to 硬风险上界"的问题形态，与 Steerable 轴"有界干预"（norm cap 内求收益、越界即 noop）在决策哲学上同构，都拒绝无约束的期望最大化；但对象（图匹配 vs 残差干预）与技术（组合近似 vs 有界低秩算子）完全不同，关联止于这一句。

### 22. Learnability of DNF with Representation-Specific Queries（ITCS 2013）

**文件**：papers/22-dnf-representation-specific-queries-itcs2013.pdf

杨柳一作（与 Blum、Carbonell），在 PAC 框架内引入**表征特定的成对查询**：给定样本中一对正例，布尔查询问两者是否在目标 DNF 中共享至少一个 term，数值查询问共享几个；动机场景是欺诈检测中专家能廉价判断"两个正例是否属于同一种欺诈"。她证明了全部正负结果：一般分布下布尔查询不比传统 PAC 容易；均匀分布下 O(log n) 相关变量、每变量至多出现 O(log n) 项、至多 2^O(√log n) 项的 DNF 可 properly learn；数值查询下均匀分布任意 DNF 可学，且若允许多例查询或自构造样例查询，则任意分布任意 DNF 可高效 properly learn。谱系意义大于内容本身：成对相似性查询是一种弱监督形式，直接把 MSU 时期度量学习的 pairwise constraints 抬进了查询复杂度理论。对 volvence 的一句话提醒：若未来引入人类成对比较协议（"这两个 turn 是否同类"），此文说明成对相似信息的价值高度依赖分布与表征假设，不能默认它比逐点标注更省。

### 26. Dynamic Matrix Factorization with Social Influence（MLSP 2016）

**文件**：papers/26-dynamic-matrix-factorization-social-mlsp2016.pdf

第三作者（Aravkin、Varshney、Yang，IBM 时期），把用户偏好的两类动态——个体随时间演化、经社交连接受他人影响——统一进 state-space 动态矩阵分解。方法上放宽了以往动态 MF 的线性/高斯限制：把 Kalman smoothing 改写为单个大规模优化问题用 quasi-Newton 求解，社交影响经 graph Laplacian 正则进入动态项；Epinions 大规模数据上 RMSE 一致下降。与 volvence 的距离：远亲。volvence 的 `user_model` 语义 owner 同样要跟踪用户状态随时间的漂移，但走的是快照化命名状态 + PE 结算，而非潜因子滤波；漂移的理论对接面在簇 3（drifting distribution / drifting target concept），此文只是应用侧佐证"偏好漂移应显式建模而非当噪声处理"这一常识。

### 27. Characterizing the Distortion of Some Simple Euclidean Embeddings（EuroCG 2016）

**文件**：papers/27-distortion-euclidean-embeddings-eurocg2016.pdf（另存于 papers/27-eurocg2016-proceedings-booklet.pdf 第 159–162 页）

Lenchner、Onak、Sheehy、Yang（IBM Watson 时期）刻画几类简单欧氏嵌入的最坏失真：圆上 N 点嵌入一条线失真 Θ(N)，嵌入两条线降到 O(√N)，三条线即可常数失真；并讨论 R^(K+1) 中 N+1 点（仅一点在超平面外）嵌入 R^K 的失真。方法是 Kirszbraun 延拓定理、Borsuk-Ulam 等经典工具的初等组合。与 volvence 的距离：远，纯计算几何登记项。可顺手记一句：允许多 chart（多条线/多个平面）后失真骤降，提示低维读出不必强求单一全局表示——但 volvence 当前的 reader 是任务驱动的判别式读出，不做等距嵌入，此观察无落点。

### 28. A Boosting Framework for Visuality-Preserving Distance Metric Learning and its Application to Medical Image Retrieval（TPAMI 2010）

**文件**：papers/28-boosting-visuality-preserving-dml-tpami2010.pdf

杨柳一作的期刊集大成之作（MSU→CMU 过渡期），指出医学图像检索中"相似"有两义——视觉外观相似与语义标注相似（两个看起来很不同的肿瘤可以都恶性）——既有 DML 只优化其一：只顾语义会检回长得不像的图让医生不信任系统，只顾外观会检回表面相似但语义无关的图诱导误诊。方法是 boosting 框架：先从成对标注学二值表示，距离取加权 Hamming，在乳腺 ISADS、ImageCLEF、COREL 上以更低计算成本比肩或超过 SOTA。与 volvence 的距离：远，但有一句真实对接——它把"表面相似 ≠ 语义相似"当一等设计约束，与 semantic-embedding-backend spec 的立场同构：语义决策必须用语义级方法，且不同相似度承担不同职责、不许一个阈值包打天下（如 CaseMemory 用独立的 structured applicability gate 而非单一 topic-similarity 阈值判断行动适用性）。历史佐证，不是机制来源。

### 29. How Much Distortion Can be Caused by One Bad Point?（FWCG 2015）

**文件**：papers/29-distortion-one-bad-point-fwcg2015.pdf

Onak、Lenchner、Yang（IBM）；实际标题用 *Caused*，论文集 txt 误作 *Incurred*。核心结果：线上奇数 N 个等距点，加上中心正上方高 √N 的一个点，任何到线的嵌入都有 Ω(√N) 失真——即嵌入质量可以被**单个**离群点毁掉，且该下界与允许任意多点在超平面外的一般情形相比并不宽松多少。方法是非收缩嵌入下的初等反证。与 volvence 的距离：远。一句话级启示：任何依赖低维投影的全局读出，其失真对单个离群样本不鲁棒是数学事实而非工程细节；volvence 的 residual reader 是判别式训练而非等距嵌入，该风险形态不同，此启示仅止于警句，不构成行动项。

### 32. Semi-supervised Learning with Weakly-Related Unlabeled Data: Towards Better Text Categorization（NIPS 2008）

**文件**：papers/32-ssl-weakly-related-unlabeled-nips2008.pdf

杨柳一作（本簇两个单点之一）。问题设定诚实而少见：当无标注语料与目标类只是**弱相关**时，主流 SSL 依赖的 cluster assumption 失效——把决策边界推向弱相关数据的低密度区不再有理由帮助分类。SSLW 的关键假设是"话题不同，但同一语言内词的用法模式跨语料一致"：估计一个同时与弱相关语料的共现信息、与标注数据一致的最优 word-correlation matrix，在最大间隔框架内做归纳式学习，凸优化近似求解；小训练集文本分类上显著优于 SOTA 归纳式 SSL。对 volvence 的远亲对应：companion 对话域数据稀缺时，能否用弱相关的非 companion 语料（一般对话、书面文本）辅助 reader/prototype 的学习？#32 的启示是别把弱相关数据当同分布数据用，而应只提取跨域稳定的低层结构（它提的是词相关，volvence 侧的对应物是表征几何）。登记为远期语料策略参照，无当前落点；去向见簇级小结。

### 33. Unifying Discriminative Visual Codebook Generation with Classifier Training for Object Category Recognition（CVPR 2008 oral）

**文件**：papers/33-unifying-codebook-classifier-cvpr2008.pdf

杨柳一作 oral（MSU 时期，与 Jin、Sukthankar、Jurie）。针对 bag-of-visual-words 的老问题——无监督聚类出的码本与分类器训练脱节，单个词判别但组合未必利于分类——提出统一优化框架：每个图像特征编码为按类别优化的"visual bits"序列，根据分类器在训练集上的表现迭代增生新 bits、再更新分类器，直至达标。与 volvence 的距离：远，但值得作为**对照系**登记一句：这是"表示与下游任务联合优化"路线的典型，而 volvence 刻意走反面——冻结基底 + lineage 冻结的固定 reader，把适应限制在有界控制器层，用联合最优性换可审计性与可回滚性（R2）。二者是设计光谱的两端；此文帮助说明 volvence 的选择是有代价的自觉选择，而非无知于联合优化。

### 34. Bayesian Active Distance Metric Learning（UAI 2007 oral）

**文件**：papers/34-bayesian-active-dml-uai2007.pdf

杨柳一作 oral（MSU，与 Jin、Sukthankar），本簇最值得写透的一篇。针对既有 DML 的两个失败模式——小样本下点估计不可靠、训练 pair 随机选取浪费标注——提出 Bayesian DML：以等价/不等价约束的 logistic 似然为基础，用变分法估计距离度量的**完整后验分布**，再把后验用于主动学习——以 Laplacian 近似计算未标注 pair 相对距离的不确定性，选**不确定性最大**的 pair 送人工标注；图像分类与口语字母识别实验中，精度与所选样本的信息量均优于非贝叶斯方法及当时 SOTA DML。在她的轨迹里，这是"标注预算应流向模型最不确定处"原则的第一次完整实现，也是 2009 年后 CMU 查询复杂度纲领（proactive learning、buy-in-bulk、自验证 AL）的应用雏形；从点估计走向"度量上的分布"，同时预示了她后来的贝叶斯主动学习与先验估计理论线（簇 1 的 #16/#24，簇 3 的 #10/#15/#21）。**落点**：直接思想对应是主线提升方案的 C2 人类验证锚——标注预算有限时"选哪些 turn 送标"，#34 的答案迁移过来即"优先送 sensor belief / gate 决策后验最不确定、或 PE 结算分歧最大的 turn"，让每份人类标注买到最大验证信息量。**边界**：迁移必须保住 volvence 的硬约束——人类标注只作验证锚（校验 reader/gate 读出质量），不得回灌为训练信号（[steering-human-anchor.md](../../docs/specs/steering-human-anchor.md)）；#34 原文是把主动选出的标注直接用于后验更新的，这一步违反 Learnable 轴"只从 PE/credit 学习"的禁令，**不迁移**。跨簇口径：本篇主评于此，查询策略的理论化由簇 1（标签复杂度：#13/#24/#30/#31）与簇 2（#04 的 mistakes-queries 权衡）承接，标注预算主题在彼处展开（见 [00_PAPER_INDEX.md](00_PAPER_INDEX.md) 簇分配说明）。

### 35. Discriminative Cluster Refinement（CVPR 2007）

**文件**：papers/35-discriminative-cluster-refinement-cvpr2007.pdf

杨柳一作（MSU，与 Jin、Pantofaru、Sukthankar）。问题：无监督聚出的视觉词丢失词间语义关系，小训练集下同义特征被量化拆进不同簇、导致类关联估不准；DCR 用视觉词的共现信息显式建模成对关系，以类标签识别对分类最有信息量的共现模式，最大间隔框架下学出最优 kernel matrix。好处是可平滑接入任何聚类方法产出的既有码本；PASCAL 2006 小样本设定下显著提升。与 volvence 的距离：远。谱系登记：其动机（量化表示丢语义关系、小样本更致命）是 #33 联合优化路线的前奏，与 #39 的"两组相似度对齐"形式化同族；对 volvence 无对接点。

### 36. Learning Distance Metrics for Interactive Search-assisted Diagnosis of Mammograms（SPIE MI 2007）

**文件**：papers/36-distance-metrics-mammograms-spie2007.pdf

杨柳一作（MSU，与 Jin、Sukthankar、UPMC 放射科合作者等）。ISAD（interactive search-assisted diagnosis）的定位是不替医生下诊断、只检回相似的已标注病例供人对照决策；本文聚焦其中被医学影像界忽视的一环——高维特征空间上的相似度本身应该被学习。给出多个 DML 算法，在 2522 个活检定标的乳腺 ROI（1800 恶性、722 良性）上验证：学到的度量同时改善分类（ROC）与检索精度；检索端由 Diamond 分布式平台支撑交互速度。与 volvence 的距离：远，历史登记：这是 #28（TPAMI 2010）的前奏与数据基础，属"人机交互回路中的度量学习"应用背景，不做机制引申。

### 37. Resource-constrained Supervised Dimensionality Reduction（IJCAI-WS MIR 2007 oral）

**无公开 PDF**（工作坊论文集未留存；依贡献声明与 CV 条目登记）

杨柳一作 oral，贡献声明仅一句："an efficient algorithm for resource-constrained supervised dimensionality reduction"。可靠可说的只有：题目落在监督降维（supp survey 分类学中与 DML 紧邻的一支），且显式带"资源受限"约束。谱系登记一句：从"资源受限"字样可见她的"预算意识"（标注预算、查询预算、计算预算）在 2007 年已是显式设计变量，这一意识后来在 CMU 被理论化为 cost complexity（簇 1 的 #30/#31）。细节无法核实，不做进一步引申。

### 38. An Efficient Algorithm for Local Distance Metric Learning（AAAI 2006 oral）

**文件**：papers/38-local-distance-metric-learning-aaai2006.pdf

杨柳一作 oral（MSU，与 Jin、Sukthankar、Yi Liu），她美国时期的起点作品。观察：类分布多峰时，全局度量的"类内紧致"与"类间可分"两目标彼此冲突、无法同时满足；LDM 改为优化**局部**紧致性与**局部**可分性。方法：概率框架下用特征值分析 + bound optimization 高效求解，分类与检索均显著优于全局度量学习和 kernel KNN。与 volvence 的距离：远。谱系登记：局部 vs 全局度量的张力是她度量学习期的入场问题（supp survey 将其归入 local adaptive 一类，#34 的 related work 亦引之为 LDM），无 volvence 对接点。

### 39. Semi-supervised Multi-label Learning by Constrained Non-negative Matrix Factorization（AAAI 2006 oral）

**文件**：papers/39-ssl-multilabel-cnmf-aaai2006.pdf

第三作者（Yi Liu、Rong Jin、Liu Yang）。针对类别数大、训练数据小的多标签学习，关键假设是"输入模式相似 ⇒ 类隶属重叠大"：分别算输入模式相似度 K_x 与类隶属相似度 K_y，寻找使两组相似度之差最小的未标注数据类隶属指派，形式化为约束 NMF 并给出高效算法；大类数小样本的文本分类上显著优于 SOTA。与 volvence 的距离：远。登记一句：其"两个相似度空间对齐"的形式化与 #35 的 kernel 优化同族，是 MSU 组把成对相似性当作统一监督货币的又一例证；无对接点。

### 40. 基于边缘匹配与多尺度小波变换的图像配准算法（华中科技大学学报·自然科学版 2004）

**无公开 PDF**（CNKI 收录；依中文贡献声明登记）

杨柳一作，本科（华中科技大学电子工程）期作品，她的科研起点。依声明：选出相对基准点位置相似且局部链码匹配的对应边缘曲线段，再用多尺度小波变换提取对应边缘曲线段上的真实角点，并在实时系统中做仿真试验——经典的边缘特征图像配准工程。与后续学术主线唯一的连线是"对应/匹配"这个母题（配准的对应点 → 度量学习的成对约束 → 表征特定成对查询），但这是叙事性巧合而非方法传承。纯历史登记，与 volvence 无关联。

## 簇级小结

C 档判词成立：16 篇没有一篇携带可直接进入 volvence 四轴的机制，本簇的价值在于解释她后期理论纲领的来源——在应用里反复撞上"标注昂贵、成对监督、预算受限"，然后回头为这堵墙建理论。两个单点的去向：**#34** 的"后验不确定性驱动标注选择"经簇 1（标签复杂度 / proactive learning）与簇 2（选择性采样的 mistakes-queries 权衡）的标注预算主题，间接服务 C2 人类验证锚的送标预算分配，且迁移时人类标注只作验证锚、不作学习源的硬边界不动；**#32** 的"弱相关无标注语料利用"登记为远期"非 companion 语料辅助"的语料策略参照，无当前落点、不设行动项。明确结论：**本簇不进入高优先级转化清单，仅 #34 的思想经簇 1/簇 2 间接进入**；其余各篇作为历史谱系与对照系（#33 联合优化 vs 冻结基底）留档即可。
