# 杨柳论文集总索引（40 篇）

> 编号 = 《副本杨柳论文集20230630V2.4.txt》中的"按重要程度"序号（作者自评）。
> 杨柳 = Liu Yang：华中科技大学电子工程本科 → MSU（Rong Jin 组，度量学习/半监督）→ CMU 机器学习系博士（2013，导师 Avrim Blum & Jaime Carbonell，论文《Mathematical Theories of Interaction with Oracles》，委员会含 Manuel Blum / Sanjoy Dasgupta / Yishay Mansour / Joel Spencer）→ CMU 博后（Blum）→ IBM Research → 中国联通数字科技。
> Steve Hanneke（Purdue，主动学习理论权威）是其最长期的合作者（16 篇合作论文）。
>
> 簇号：C1 = 主动学习标签复杂度（[01](01_ACTIVE_LEARNING_LABEL_COMPLEXITY.md)）；C2 = RL/学徒学习/择时查询（[02](02_RL_APPRENTICESHIP_SELECTIVE_SAMPLING.md)）；C3 = 迁移先验/漂移/非平稳（[03](03_TRANSFER_PRIOR_NONSTATIONARY.md)）；C4 = 测试与审计理论（[04](04_TESTING_AUDIT_THEORY.md)）；C5 = 早期 ML 与 CS 理论杂项（[05](05_EARLY_ML_AND_THEORY_MISC.md)）。
> 档位：A = 直接进入四轴整合清单；B = 思想级参照；C = 背景登记。详见各簇文档与 [06_VZ_INTEGRATION_FOUR_AXES.md](06_VZ_INTEGRATION_FOUR_AXES.md)。
> 档位口径：以各簇文档**深读后的逐篇判定为准**。深读相对初始预估下调 5 篇：#17/#24/#30/#31 A→B（思想教训/理论根/远期骨架可用，但无近期可跑转写）、#02 B→C（无公开版，仅依贡献声明保守分析）。最终分布：**A×16、B×12、C×12**。

| # | 论文 | 发表 | 年份 | 作者序/贡献 | 簇 | 档 | PDF | 来源 |
|---|---|---|---|---|---|---|---|---|
| 01 | Bandit Learnability can be Undecidable | COLT | 2023 | 第二（字母序）；建立 bandit 可学习性一般理论，证明其在 ZFC 内不可判定；用 teaching dimension 刻画二值奖励 bandit 最优查询复杂度 | C2 | A | `01-bandit-learnability-undecidable-colt2023.pdf` | Hanneke 主页 |
| 02 | Active Learning with Identifiable Mixture Models | Annals of Statistics（投稿中） | 2023 | 第一（打破字母序）；参数混合模型下主动学习风险一般上界；指数族混合特例 | C1 | C | **无公开版**（Hanneke 主页仍标 in preparation） | 引用登记 |
| 03 | Reliable Active Apprenticeship Learning | ALT | 2025 | 第二；开创方向；无噪声情形主算法与正确性；Eluder-star dimension；Massart/Tsybakov 噪声下算法与上下界；agnostic 设定 | C2 | A | `03-reliable-active-apprenticeship-learning-alt2025.pdf` | PMLR v272 |
| 04 | Toward a General Theory of Online Selective Sampling: Trading Off Mistakes and Queries | AISTATS | 2021 | 第二（字母序）；在线选择性采样一般理论框架；mistakes-queries 最优权衡曲线性质 | C2 | A | `04-online-selective-sampling-aistats2021.pdf` | Hanneke 主页 |
| 05 | Computing and Testing Small Connectivity in Near-Linear Time and Queries via Fast Local Cut Algorithms | SODA | 2020 | 第三（字母序）；O(k²ν) 局部割检测随机算法；k-边/点连通性质测试近线性查询复杂度（解决两个 20 年开放问题） | C4 | C | `05-small-connectivity-local-cut-soda2020.pdf` | arXiv:1910.14344 |
| 06 | Statistical Learning under Nonstationary Mixing Processes | AISTATS | 2019 | 第二（字母序）；非平稳 β-mixing 过程学习算法；有界 VC subgraph 类累积超额风险次线性增长 | C3 | A | `06-nonstationary-mixing-processes-aistats2019.pdf` | Hanneke 主页 |
| 07 | Surrogate Losses in Passive and Active Learning | EJS | 2019 | 第二（字母序）；基于 classification-calibrated 代理损失的主动学习算法与标签请求数分析 | C1 | A | `07-surrogate-losses-passive-active-ejs2019.pdf` | Hanneke 主页 |
| 08 | A Theory of Transfer Learning with Applications to Active Learning | Machine Learning | 2013 | 第一（打破字母序）；提出设定：目标概念序列从未知先验独立采样；渐近标签复杂度界 | C3 | A | `08-theory-transfer-learning-active-ml2013.pdf` | Hanneke 主页 |
| 09 | Minimax Analysis of Active Learning | JMLR | 2015 | 第二（字母序）；发现 star number 组合复杂度度量；证明几乎所有既有 AL 复杂度度量最坏值等于 star number | C1 | A | `09-minimax-analysis-active-learning-jmlr2015.pdf` | Hanneke 主页 |
| 10 | Identifiability of Priors from Bounded Sample Sizes with Applications to Transfer Learning | COLT | 2011 | 第一（打破字母序）；开创题目；有界样本先验可辨识性；渐近标签复杂度界 | C3 | A | `10-identifiability-priors-transfer-colt2011.pdf` | Hanneke 主页 |
| 11 | Active Learning with a Drifting Distribution | NIPS | 2011 | 唯一作者；流式设定分布随时间漂移；disagreement-based AL 的错误数与查询数上界 + minimax 下界 | C3 | A | `11-active-learning-drifting-distribution-nips2011.pdf` | 杨柳主页 |
| 12 | Learning with a Drifting Target Concept | ALT | 2015 | 第三（字母序）；目标概念每步漂移 ≤Δ 时的误差界 Õ(√d√Δ)；均匀分布线性分类器多项式时间算法改进 | C3 | A | `12-learning-drifting-target-concept-alt2015.pdf` | Hanneke 主页 |
| 13 | Buy-in-Bulk Active Learning | NIPS | 2013 | 第一（打破字母序）；批量标签请求的标签复杂度界；总查询数与轮数的权衡 | C1 | A | `13-buy-in-bulk-active-learning-nips2013.pdf` | NeurIPS 官方 |
| 14 | Active Property Testing | FOCS | 2012 | 第四（字母序）；提出 testing dimension 刻画测试固有查询数；区间并/高斯分布线性分类器测试算法与下界 | C4 | A | `14-active-property-testing-focs2012.pdf` | arXiv:1111.0897 |
| 15 | Bounds on the Minimax Rate for Estimating a Prior over a VC Class from Independent Learning Tasks | ALT | 2015 | 第一（打破字母序）；从独立任务序列估计 VC 类先验的最优收敛速率 | C3 | A | `15-prior-estimation-vc-minimax-alt2015.pdf` | Hanneke 主页 |
| 16 | Bayesian Active Learning Using Arbitrary Binary Valued Queries | ALT | 2010 | 第一（打破字母序）；任意是/否问题下达到指定风险的期望查询数界（率失真视角） | C1 | B | `16-bayesian-al-binary-queries-alt2010.pdf` | 杨柳主页 |
| 17 | Activized Learning with Uniform Classification Noise | ICML | 2013 | 第一（打破字母序）；均匀分类噪声下任意被动算法可转化为主动算法并强渐近降低标签复杂度 | C1 | B | `17-activized-learning-uniform-noise-icml2013.pdf`（附 `supp-activized-icml2013-supplemental.pdf`） | Hanneke 主页 |
| 18 | Online Learning by Ellipsoid Method | ICML | 2009 | 第一；椭球逼近版本空间的在线学习算法；质心与正定矩阵高效更新；USPS/UCI 实验 | C2 | B | `18-online-learning-ellipsoid-icml2009.pdf` | ICML 官方归档 |
| 19 | Online Allocation and Pricing with Economies of Scale | WINE | 2015 | 第三（字母序）；边际成本递减商品在线分配贪心算法常数近似；样本复杂度分析 | C5 | C | `19-online-allocation-economies-scale-wine2015.pdf` | 杨柳主页 |
| 20 | Risk-Averse Matchings over Uncertain Graph Databases | ECML PKDD | 2018 | 第四（字母序）；不确定加权（超）图上最大期望奖励+有界风险匹配，NP-hard，Ω(1/k) 近似算法 | C5 | B | `20-risk-averse-matchings-ecml2018.pdf` | arXiv:1801.03190 |
| 21 | Bounds on the Minimax Rate…（#15 期刊版） | TCS | 2018 | 第一（打破字母序）；#15 的期刊扩展：光滑先验条件下的最优速率 | C3 | A | `21-prior-estimation-vc-minimax-tcs2018.pdf` | Hanneke 主页 |
| 22 | Learnability of DNF with Representation-Specific Queries | ITCS | 2013 | 第一（打破字母序）；成对布尔/数值查询下 PAC 学习 DNF 的全部正负结果 | C5 | B | `22-dnf-representation-specific-queries-itcs2013.pdf` | 杨柳主页 |
| 23 | Testing Piecewise Functions | TCS | 2018 | 第二（字母序）；零测度交叉条件下实线上一般分段函数性质测试查询复杂度；主动测试下与段数无关 | C4 | A | `23-testing-piecewise-functions-tcs2018.pdf` | Hanneke 主页 |
| 24 | The Sample Complexity of Self-Verifying Bayesian Active Learning | AISTATS | 2011 | 第一（打破字母序）；先验知识使自终止主动学习样本复杂度恒优于先验相关被动学习 | C1 | B | `24-self-verifying-bayesian-al-aistats2011.pdf` | Hanneke 主页 |
| 25 | Negative Results for Active Learning with Convex Losses | AISTATS | 2010 | 第二（字母序）；凸损失 proper 主动学习即使有界噪声下 minimax 速率常不优于被动学习 | C1 | A | `25-negative-results-convex-losses-aistats2010.pdf` | Hanneke 主页 |
| 26 | Dynamic Matrix Factorization with Social Influence | MLSP | 2016 | 第三（字母序）；个体演化+社交影响双动态过程模型；动态矩阵分解高效估计 | C5 | C | `26-dynamic-matrix-factorization-social-mlsp2016.pdf` | arXiv:1604.06194 |
| 27 | Characterizing the Distortion of Some Simple Euclidean Embeddings | EuroCG | 2016 | 第四（字母序）；圆上 N 点嵌入两条线 O(√N) 失真、三条线常数失真等 | C5 | C | `27-eurocg2016-proceedings-booklet.pdf` 第 159–162 页 | EuroCG 官方论文集 |
| 28 | A Boosting Framework for Visuality-Preserving Distance Metric Learning and its Application to Medical Image Retrieval | TPAMI | 2010 | 第一；同时保持视觉与语义相似性的 boosting 度量学习框架；乳腺影像/ImageCLEF/COREL 评估 | C5 | B | `28-boosting-visuality-preserving-dml-tpami2010.pdf` | Satyanarayanan 主页 |
| 29 | How Much Distortion Can be Incurred from One Bad Point? | FWCG | 2015 | 第三（字母序）；线上奇数 N 点+一个高 √N 坏点嵌入线的 Ω(√N) 失真下界等 | C5 | C | `29-distortion-one-bad-point-fwcg2015.pdf` | FWCG2015 官方 |
| 30 | Cost Complexity of Proactive Learning via a Reduction to Realizable Active Learning | CMU-ML-09-113 | 2009 | 第一（打破字母序）；主动学习的 oracle 选择+实例选择成本-可靠性权衡一般框架；两类序贯假设检验 | C1 | B | `30-proactive-learning-cost-complexity-cmu-ml-09-113.pdf` | CMU 报告库 |
| 31 | Adaptive Proactive Learning with Cost-Reliability Tradeoff | CMU-ML-09-114 | 2009 | 第一（打破字母序）；proactive learning 理论框架；任意噪声多持久 oracle 元过程 | C1 | B | `31-adaptive-proactive-learning-cmu-ml-09-114.pdf` | CMU 报告库 |
| 32 | Semi-supervised Learning with Weakly-Related Unlabeled Data: Towards Better Text Categorization | NIPS | 2008 | 第一；弱相关无标注数据的归纳式半监督学习（最大间隔+词相关矩阵） | C5 | B | `32-ssl-weakly-related-unlabeled-nips2008.pdf` | NeurIPS 官方 |
| 33 | Unifying Discriminative Visual Codebook Generation with Classifier Training for Object Category Recognition | CVPR | 2008 | 第一（oral）；判别式视觉码本生成与分类器训练统一优化框架 | C5 | C | `33-unifying-codebook-classifier-cvpr2008.pdf` | 杨柳主页 |
| 34 | Bayesian Active Distance Metric Learning | UAI | 2007 | 第一（oral）；度量后验分布估计；选择相对距离不确定性最大的未标注样本对 | C5 | B | `34-bayesian-active-dml-uai2007.pdf` | 杨柳主页 |
| 35 | Discriminative Cluster Refinement: Improving Object Category Recognition Given Limited Training Data | CVPR | 2007 | 第一；视觉词共现建模+最大间隔核矩阵优化 | C5 | C | `35-discriminative-cluster-refinement-cvpr2007.pdf` | 杨柳主页 |
| 36 | Learning Distance Metrics for Interactive Search-assisted Diagnosis of Mammograms | SPIE MI | 2007 | 第一；交互式搜索辅助诊断的度量学习算法；2522 乳腺 ROI 评估 | C5 | C | `36-distance-metrics-mammograms-spie2007.pdf` | 杨柳主页 |
| 37 | Resource-constrained Supervised Dimensionality Reduction | IJCAI-WS MIR | 2007 | 第一（oral）；资源受限监督降维高效算法 | C5 | C | **无公开版** | 引用登记 |
| 38 | An Efficient Algorithm for Local Distance Metric Learning | AAAI | 2006 | 第一（oral）；局部紧致性+局部可分性优化；特征值分析+bound optimization | C5 | B | `38-local-distance-metric-learning-aaai2006.pdf` | 杨柳主页 |
| 39 | Semi-supervised Multi-label Learning by Constrained Non-negative Matrix Factorization | AAAI | 2006 | 第三（oral）；两组相似度差最小化的约束 NMF 形式化 | C5 | C | `39-ssl-multilabel-cnmf-aaai2006.pdf` | AAAI CDN |
| 40 | 基于边缘匹配与多尺度小波变换的图像配准算法 | 华中科技大学学报（自然科学版） | 2004 | 第一；边缘曲线段匹配+多尺度小波真实角点提取；实时系统仿真 | C5 | C | **无公开版**（CNKI 收录） | 引用登记 |

## 补充材料（非 40 篇清单内）

| 文件 | 说明 |
|---|---|
| `supp-phd-thesis-oracles-cmu2013.pdf` | 博士论文《Mathematical Theories of Interaction with Oracles》（CMU 2013，2.1MB）——统一其主动学习/先验估计/DNF 查询/自验证等主线的总纲，四个核心簇的最佳伴读材料 |
| `supp-cv-liu-yang-2014.pdf` | 2014 版 CV（作者履历与研究陈述佐证） |
| `supp-buy-in-bulk-techreport-cmu-ml-12-110.pdf` | #13 的 CMU 技术报告版（更完整证明） |
| `supp-lossy-coding-journal-manuscript.pdf` | #16 的信息论表述期刊稿《Characterizing Optimal Rates for Lossy Coding with Finite-Dimensional Metrics》 |
| `supp-dml-comprehensive-survey-msu2006.pdf` | 《Distance Metric Learning: A Comprehensive Survey》（MSU 2006）——C5 簇度量学习背景 |
| `supp-activized-icml2013-supplemental.pdf` | #17 的 ICML 补充材料（完整证明） |

## 不可下载项引用登记（3 篇）

1. **#02** Yang, L., Hanneke, S., Castelli, V. *Active Learning with Identifiable Mixture Models*. In submission to Annals of Statistics（2023）。无公开预印本；分析依据：论文集贡献声明 + Hanneke 主页 in-preparation 条目。
2. **#37** Yang, L., Jin, R., Sukthankar, R. *Resource-constrained Supervised Dimensionality Reduction*. The First International Workshop on Multimodal Information Retrieval at IJCAI, 2007（oral）。工作坊未留存公开论文集；分析依据：贡献声明 + CV 条目。
3. **#40** 杨柳等. 基于边缘匹配与多尺度小波变换的图像配准算法. 华中科技大学学报（自然科学版），2004 年 11 月。CNKI 收录无公开 PDF；分析依据：贡献声明（中文原文）。

## 40 篇 → 簇分配汇总

- **C1 主动学习标签复杂度（11 篇）**：02, 07, 09, 13, 16, 17, 24, 25, 30, 31 + 34（跨簇：主动度量学习，简评在 C5、查询策略结合点在 C1 引用）→ 正式计入 C1 的为 02/07/09/13/16/17/24/25/30/31（10 篇）+ 跨簇引用 34
- **C2 RL/学徒学习/择时查询（4 篇）**：01, 03, 04, 18
- **C3 迁移先验/漂移/非平稳（7 篇）**：06, 08, 10, 11, 12, 15, 21
- **C4 测试与审计理论（3 篇）**：05, 14, 23
- **C5 早期 ML 与 CS 理论杂项（15 篇）**：19, 20, 22, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40（16 项——34 主评在此，故 C1 实收 10 篇）

> 口径说明：C1=10 正式 + 1 跨簇引用（34），C5=16 篇简评（含 34），合计 10+4+7+3+16 = 40。
