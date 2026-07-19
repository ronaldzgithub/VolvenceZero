# 论文全览 — 8 轴 26 篇

每条格式：**标题**（年份）— 一句话定位。本地 PDF / 链接。

## 轴 1：连接组测绘（Connectome）

1. **A reference brain for the clonal raider ant**（2025, bioRxiv/Current Biology）— 40 个克隆蚂蚁个体做出首个蚁脑参考图谱，意外发现基因/年龄/形态完全相同的个体间脑结构差异巨大，猜测与分工相关。
   [`papers/connectome/reference-brain-clonal-raider-ant-2025.pdf`](papers/connectome/reference-brain-clonal-raider-ant-2025.pdf) · https://doi.org/10.1016/j.cub.2025.11.018

2. **CRANTb / CRANTpy — Clonal Raider ANT Brain 突触级连接组**（2025）— 继 FlyWire 果蝇之后第二个突触级全脑昆虫连接组，Python/R 包可编程查询。
   https://social-evolution-and-behavior.github.io/crantpy/intro.html

3. **Mapping the Brain of the Clonal Raider Ant**（2025, PhD thesis, Rockefeller）— 整合行为/解剖/连接组的博士论文，聚焦嗅觉系统。
   https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/813/

4. **Sparse and stereotyped encoding implicates a core glomerulus for ant alarm behavior**（2023, Cell）— ~500 个嗅觉小球中，警报信息素只激活 ≤6 个，且收敛到单一"panic glomerulus"枢纽，说明高风险快速反应走的是**精确、定型的硬编码式**通路而不是组合编码。
   全文已读，仅链接：https://doi.org/10.1016/j.cell.2023.05.025

5. **Pheromone representation in the ant antennal lobe changes with age**（2024, bioRxiv）— 同一批小球对警报信息素的敏感度随年龄重新加权，年长蚂蚁的核心枢纽小球响应更强，对应其更多参与巢防御。
   全文已读，仅链接：https://doi.org/10.1101/2024.02.13.580193

## 轴 2：菌体（Mushroom Body）计算模型

6. **Using an Insect Mushroom Body Circuit to Encode Route Memory in Complex Natural Environments**（Ardin et al., 2016, PLOS Comp Biol）— 果蝇菌体的 spiking 模型可以直接搬来解释沙漠蚁的视觉路线记忆；稀疏码 + 一次性学习，估算存储容量约几百张独立图像。
   [`papers/mushroom-body/ardin-mb-route-memory-2016.pdf`](papers/mushroom-body/ardin-mb-route-memory-2016.pdf)

7. **Investigating visual navigation using spiking neural network models of the insect mushroom bodies**（2024, Frontiers in Physiology）— SNN 菌体模型 + 机器人实体化验证，研究稀疏度/曝光时长/训练集大小对 Kenyon cell 学习动态的影响。
   [`papers/mushroom-body/snn-mb-visual-navigation-2024.pdf`](papers/mushroom-body/snn-mb-visual-navigation-2024.pdf)

8. **Latent learning without map-like representation of space in navigating ants**（2024, bioRxiv）— 蚂蚁没有全局地图，靠连续更新的潜在记忆 + 动态门控机制做逐时刻自我中心决策；多个 MBON 构成并行记忆库（短/长时、趋近/回避）。
   [`papers/mushroom-body/latent-learning-no-map-ants-2024.pdf`](papers/mushroom-body/latent-learning-no-map-ants-2024.pdf)

## 轴 3：中央复合体（Central Complex）/ 环形吸引子

9. **The head direction circuit of two insect species**（2020, PMC）— 用解剖投影数据构建环形吸引子模型，预测跨物种连接强度差异下仍能维持朝向编码。
   全文已读，仅链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC7419142/

10. **Emergent spatial goals in an integrative model of the insect central complex**（2024, PLOS Comp Biol）— CX 连接性支持路径积分 + 向量导航（home vector），期望朝向的编码方式支持向量旋转/加法运算。
    [`papers/central-complex/emergent-spatial-goals-cx-2024.pdf`](papers/central-complex/emergent-spatial-goals-cx-2024.pdf)

11. **How the insect central complex could coordinate multimodal navigation**（eLife 73077）— copy-and-shift 机制 + 环形吸引子整合多模态方向线索，steering 电路计算当前朝向与期望朝向之差。
    [`papers/central-complex/multimodal-navigation-cx-coordination.pdf`](papers/central-complex/multimodal-navigation-cx-coordination.pdf)

12. **A computational model for angular velocity integration in a locust heading circuit**（2024, PLOS Comp Biol）— 蝗虫朝向环路的角速度积分模型，乘法式调制而非加法式。
    [`papers/central-complex/locust-heading-angular-velocity-2024.pdf`](papers/central-complex/locust-heading-angular-velocity-2024.pdf)

13. **Theoretical principles explain the structure of the insect head direction circuit**（2024, eLife 91533）— 从头推导 8 列正弦式连接权重是朝向编码抗噪的最优解，并证明可通过 Hebbian 学习自然涌现。
    全文已读，仅链接：https://doi.org/10.7554/elife.91533

## 轴 4：集体智能 / 超个体（Collective Intelligence）

14. **The Bayesian Superorganism: Collective Probability Estimation in Swarm Systems**（2019, bioRxiv/ISAL）— 蚁群选巢址行为等价于近似贝叶斯计算 + 粒子滤波，无需中央协调。
    [`papers/collective-intelligence/bayesian-superorganism-swarm-2019.pdf`](papers/collective-intelligence/bayesian-superorganism-swarm-2019.pdf)

15. **The Bayesian superorganism: externalized memories facilitate distributed sampling**（2020, J R Soc Interface）— 蚂蚁避开自己/同伴留下的信息素轨迹，等价于给 MCMC 采样加"避免重复访问"记忆，显著提升采样效率。
    全文已读，仅链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC7328406/

16. **Active Inferants: An Active Inference Framework for Ant Colony Behavior**（2021, Frontiers in Behavioral Neuroscience）— 用主动推理框架统一解释蚁群觅食/T-迷宫交替行为，个体认知与信息素介导的群体认知是同一框架的两个尺度。
    [`papers/collective-intelligence/active-inferants-ant-colony-2021.pdf`](papers/collective-intelligence/active-inferants-ant-colony-2021.pdf)

## 轴 5：分工与脑可塑性（Caste Plasticity）— 反证主源

17. **Transcriptomic analysis of mosaic brain differentiation underlying complex division of labor**（2023, J Comp Neurol, Atta cephalotes）— 工蚁体型/行为/神经解剖三者分化对应的脑基因表达差异，部分模式独立于体型，直接与劳动分工挂钩。
    全文已读，仅链接：https://doi.org/10.1002/cne.25469

18. **Neuropeptides specify and reprogram division of labor in the leafcutter ant Atta cephalotes**（2024, bioRxiv）— 两种神经肽的基因敲低/注射就能让工蚁在"切叶"和"育幼"两种行为程序间**强制切换**，伴随转录组整体转向对应亚种的表达模式。
    [`papers/caste-plasticity/neuropeptides-division-of-labor-atta-2024.pdf`](papers/caste-plasticity/neuropeptides-division-of-labor-atta-2024.pdf)

19. **Epigenetic (re)programming of caste-specific behavior in the ant Camponotus floridanus**（2016, PNAS）— 组蛋白去乙酰化酶抑制剂或 RNAi 敲低 Rpd3，在羽化后立即给药可让"大型工蚁"获得"小型工蚁"的觅食/侦察行为，效果随年龄增长迅速衰减（关键窗口期）。
    全文已读，仅链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC5057185/

20. **Tramtrack acts during late pupal development to direct ant caste identity**（2021, PLOS Genetics）— 转录因子 tramtrack 在蛹期晚期就已经预先给不同亚种"标记"好未来的行为基因表达模式，是激素信号的下游执行者。
    [`papers/caste-plasticity/tramtrack-ant-caste-identity-2021.pdf`](papers/caste-plasticity/tramtrack-ant-caste-identity-2021.pdf)

## 轴 6：神经调质（Neuromodulation）

21. **Distinct mechanisms mediate dopamine-octopamine opponency in an insect model of olfaction**（2025, bioRxiv）— 多巴胺和章鱼胺通过两种不同机制（抑制性神经元的去抑制 vs 内在兴奋性调节）产生相反的嗅觉行为效应，构成一对"opponent process"状态开关。
    [`papers/neuromodulation/dopamine-octopamine-opponency-olfaction-2025.pdf`](papers/neuromodulation/dopamine-octopamine-opponency-olfaction-2025.pdf)

22. **Biogenic amines and division of labor in honey bee colonies**（Schulz & Robinson, 1999 前后系列）— 章鱼胺水平是触发/维持觅食行为状态的因果因子，与分工调控直接相关。
    综述性引用，未单独下载。

## 轴 7：连接组充分性批判（Connectome-Critique）— 反证主源

23. **OpenWorm 十年停滞的分析**（LessWrong / 多篇科普综述, 2024-2025）— C. elegans 302 个神经元的完整连接组早在 1986 年就测出，但十几年过去仍未能仅从连接组重建出真实行为；缺失突触权重、正负号、神经肽广播、跨突触信号。
    全文已读，仅链接：https://en.wikipedia.org/wiki/OpenWorm

24. **Optimization of connectome weights for a neural network model generating both forward and backward locomotion in C. elegans**（2026, Scientific Reports）— 直接用解剖连接权重跑不出行为，必须联合优化权重才能匹配行为，且发现不依赖内在起搏神经元也能生成振荡步态——说明"结构 + 猜权重"路线本质上是在拟合行为，不是在验证结构。
    [`papers/connectome-critique/connectome-weight-optimization-celegans-2026.pdf`](papers/connectome-critique/connectome-weight-optimization-celegans-2026.pdf)

## 轴 8：微型化物理极限（Miniaturization）

25. **Constant neuropilar ratio in the insect brain**（2020, Scientific Reports）— 神经元本体大小受细胞核物理尺寸限制，是昆虫体型微型化的硬瓶颈；即使最小昆虫，神经纤维网/细胞体比例仍恒定在 3:2。
    [`papers/miniaturization/constant-neuropilar-ratio-insect-brain-2020.pdf`](papers/miniaturization/constant-neuropilar-ratio-insect-brain-2020.pdf)

26. **Extremely small wasps independently lost the nuclei in the brain neurons of at least two lineages**（2023, Scientific Reports）— *Megaphragma* 寄生蜂仅 7,400 个神经元，成年后 95% 神经细胞无核，靠蛹期合成的蛋白质撑完 5 天寿命——展示了"用完即弃、不再学习"的极端资源受限策略。
    [`papers/miniaturization/anucleate-neurons-microwasps-2023.pdf`](papers/miniaturization/anucleate-neurons-microwasps-2023.pdf)

## 交叉引用：全脑仿真先例（非蚂蚁，但是"数字蚂蚁"可行性的直接参照系）

- **FlyWire 完整果蝇脑连接组**（2024, Nature 两篇 + Immersive 专题）— 139,255 个神经元、5000 万突触的完整连接组，标注 > 8,000 细胞类型。
- **A Drosophila computational brain model reveals sensorimotor processing**（Shiu et al., 2024, Nature）— 基于连接组 + 神经递质身份构建 leaky-integrate-and-fire 全脑模型，成功预测味觉/梳理回路的激活模式，笔记本电脑可跑。
- **Embodied Drosophila — Whole-Brain Connectome Simulation in a Biomechanical Body**（2025, 开源项目）— 138,639 个 LIF 神经元 + NeuroMechFly v2 physics body（MuJoCo），视觉/嗅觉/味觉/飞行全部"从连接组涌现"，无硬编码 if-else 行为规则。
  https://erojasoficial-byte.github.io/fly-brain/

## 轴 9：机器人学工程验证（Robotics）

27. **AntBot: an ant-inspired celestial compass applied to autonomous outdoor robot navigation**（Dupeyroux et al., 2019, Robotics and Autonomous Systems）— 六足机器人复刻沙漠蚁路径积分：紫外偏振光罗盘 + 步数/视觉光流测距，14 米行程后定位误差仅 5-7 cm，比民用 GPS 精度高近百倍；导航逻辑是显式向量运算，不依赖深度学习或神经形态硬件。
    全文已读，仅链接：https://doi.org/10.1016/j.robot.2019.04.007
