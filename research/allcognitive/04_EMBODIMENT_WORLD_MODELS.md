# 第四卷：具身、世界模型与潜在动作

日期：2026-07-20  
范围：19 篇主归属论文；按 `00_METHOD.md` 九项模板逐篇核读本地 PDF。

## 0. 卷首裁决

这一轴不存在单一“最佳世界模型”。证据支持把问题拆成四种不同写面：

1. **冻结特征世界模型**（DINO-WM）最适合做低成本、可规划、任务无关的数字蚂蚁主 baseline，但前提是离线轨迹覆盖充分，且目标能表达为视觉状态。
2. **在线学得的 decoder-free latent world model**（NE-Dreamer）最适合检验部分可观测、长时依赖下的表征学习；它并不等于 frozen substrate，因为 encoder、RSSM 与控制器仍共同在线更新。
3. **像素 / 视频世界模型**（NWM、HMA）提供最强可视化与数据生成上界，但生成逼真不等于动作可控、物理正确或适合作为闭环规划器。
4. **潜在动作与跨形态 VLA**（LAPA、LAOM、HPT、CrossFormer、Octo 等）证明弱标签和异构数据有价值，同时也给出核心反证：视觉变化中的 distractor 会污染 latent action；跨 embodiment 成功通常来自共享 trunk 加身体专属接口，而非一个无条件通用动作空间。

对 VZ / 数字蚂蚁的总裁决是：**冻结感知基底、身体专属动作接口、可证伪的 latent controllability probe、闭环 matched baseline 和明确 kill condition 必须同时存在。** 不能把视频质量、线性 probe、离线 action loss 或空间 QA 当作闭环控制证据。

---

## 1. 核心 sweep

### 1.1 NE-Dreamer — Next Embedding Prediction Makes World Models Stronger（2603.02765）

1. **论文事实**：George Bredis 等，1T-Tech，2026 年 3 月预印本。12M 参数；DMLab Rooms 4 任务、每法 50M 环境步、5 seeds；DMC 20 任务、每法 1M 步、5 seeds。
2. **核心问题**：decoder-free world model 若只做同时间步对齐，能否在部分可观测环境中形成真正跨时间可预测的状态，而不是漂移或坍缩。
3. **机制拆解**：图像 `x_t→e_t`，RSSM 状态 `(h_t,z_t)` 接动作；两层、4 头 causal transformer 从历史预测 stop-gradient 的 `e_{t+1}`，以 Barlow Twins 对齐；同时预测 reward/continuation。训练写 encoder、RSSM、预测器、actor/critic；推理从 RSSM prior 做 15 步 latent imagination，不写像素。
4. **关键证据**：DMLab 四任务均显著胜 DreamerV3、R2-Dreamer、DreamerPro；去掉 causal transformer 或 next-step shift 几乎失去收益，去 projector 只影响速度/稳定性；DMC 与强基线持平。post-hoc decoder 仅是相关性诊断，不是机制证明。
5. **确证价值**：强支持 P2（latent controller）与 R3/R4；中等支持 R2 中“避免像素重建负担”的方向，但**不支持 frozen substrate**，因 encoder 与控制器共同在线训练。
6. **反证价值**：反证“去掉 decoder 就足够”和“同帧自监督自然产生长时状态”；也反证像素重建是 world model 的必要条件。
7. **局部可借算法**：进入数字蚂蚁 shadow benchmark：固定参数量与步数，对比 next-embedding、same-step alignment、pixel reconstruction、reward-only；保留 causal-mask 与 target-shift 消融。
8. **不可外推边界**：DMC 接近饱和；DMLab 是模拟像素环境；未测 distractor、跨身体、真实机器人或安全在线更新；reward/continuation 仍让表示带任务信息。
9. **成熟度与裁决**：**A（主链证据）**。作为“在线 decoder-free world model”主 baseline，不可称为 frozen-feature baseline。

### 1.2 Adaptive Control via Real-Time Recurrent RL（2602.02236）

1. **论文事实**：Julian Lemmel、Felix Resch、Mónika Farsang、Ramin Hasani、Daniela Rus、Radu Grosu；TU Wien、MIT CSAIL、Liquid AI，2026 预印本。CarRacing：3 条、每条 1000 帧的人类示范，3 种 RNN×5 seeds；实机 1:10 RoboRacer：500,772 个 100Hz event frames。
2. **核心问题**：预训练 recurrent controller 能否在部署时每一步在线更新，适应传感器与动力学 shift，而不做 BPTT 或 replay-batch 更新。
3. **机制拆解**：RGB/event frame 经 CNN；CT-RNN、LRU 或 nonlinear diagonal LrcSSM 分别作为 actor/critic；TD(λ)+eligibility trace，RTRL/RFLO 前向更新；动作是连续转向/油门等。训练先 BC+重建，推理时每步写 actor 与 critic 参数，并以距预训练权重的 L2 项约束。
4. **关键证据**：CarRacing 10 圈后各模型改善，LrcSSM 最快最稳；实机 line following 10 圈后多数模型超过初始策略；0.1 rad（约动作范围 19%）转向偏置与阳光干扰后性能先降后恢复。LRU 在 event task 预训练即失败；PPO 对照结果混合。实机有人工干预，且作者明确警告不用于安全关键系统。
5. **确证价值**：支持 P2、R1/R2 的 fast adaptive controller；对“稳定基底+有界在线层”只给部分证据，因为论文实际更新主策略权重，而非严格 adapter owner。
6. **反证价值**：反证固定部署策略可覆盖非平稳分布；也暴露逐步在线 RL 会因坏超参被动退化乃至停驶。
7. **局部可借算法**：只借 forward eligibility trace、参数漂移预算、shift/recovery 曲线与停更门；数字蚂蚁中应限制为 controller-local adapter，不更新 frozen substrate。
8. **不可外推边界**：仅驾驶类任务；实机样本少且有人接管；reward 依赖 SLAM/人工设计；缺少安全约束、回滚与强 replay-based matched baseline。
9. **成熟度与裁决**：**B（强基线 / 风险反例）**。作为 online-fast adaptation 上界和 kill-condition 来源，不直接进入生产主链。

### 1.3 MEM — Multi-Scale Embodied Memory for VLA（2603.03596）

1. **论文事实**：Marcel Torne、Karl Pertsch 等，Physical Intelligence、Stanford、UC Berkeley、MIT，2026 预印本。基于 Gemma3-4B、SigLIP-400M 与 860M action expert；最长 15 分钟任务；长任务每策略/任务或菜谱 10 rollouts；42 个训练菜谱、5 个未见厨房菜谱。
2. **核心问题**：单一稠密视觉历史既昂贵又不能表达十分钟级语义进度；机器人记忆是否应按时间尺度与模态拆分。
3. **机制拆解**：低层 `π_LL` 消费最多 18 帧/54 秒视频、proprio 与语言子任务，输出连续 action chunk；高层 `π_HL` 读当前观察、目标和自回写语言摘要，输出下个子任务及更新摘要。video encoder 交替空间与 causal temporal attention，只向 VLA 传当前时刻 token；语言摘要由 LLM 生成标签训练。
4. **关键证据**：recipe setup、clean kitchen 中 video-only、text-only、naive text+video 均弱于完整 MEM；去压缩的指令拼接受 train/inference shift 影响；短时记忆使冰箱开门成功提升 62 个百分点、筷子抓取提升 11 点；pool/proprio memory 在遮挡、计数、环境状态任务失败；预训练 memory 胜 post-train-only。
5. **确证价值**：强支持 P4 与 R1/R5/R6 的多时间尺度分层；支持“谁拥有摘要谁发布”的方向，但论文的语言摘要由高层 VLA 自回写，尚非不可变 snapshot SSOT。
6. **反证价值**：反证“把所有历史塞进 context”及“一种记忆模态足够”；也显示自然语言压缩会主动丢弃失败尝试，可能掩盖关键因果证据。
7. **局部可借算法**：数字蚂蚁采用短时稠密传感 cache + 长时结构化事件 snapshot 的 matched memory baseline；必须另保留失败 provenance，不照搬可变自然语言摘要。
8. **不可外推边界**：数据规模与完整训练混合未充分公开；主要是厨房操作；高层语言 memory 有 hallucination、语义漂移与自回写误差；没有跨 episode 长期记忆。
9. **成熟度与裁决**：**A（主链证据）**。借时间尺度分解与消融，不借“Markdown 即状态”。

---

## 2. 冻结特征、像素世界模型与潜在动作

### 2.1 DINO-WM（2411.04983）

1. **论文事实**：Gaoyue Zhou、Hengkai Pan、Yann LeCun、Lerrel Pinto；NYU、Meta AI；2024/2025 预印本。6 套环境；数据从 1,000×20 步到 PushT 20,000×100 步；50 个初始/目标对（Rope/Granular 各 10）。
2. **核心问题**：能否在离线、无 reward、无 expert task demo 下，以冻结通用视觉特征学习 task-agnostic dynamics 并测试时规划。
3. **机制拆解**：冻结 DINOv2 patch encoder；19M ViT transition 读过去 patch、proprio 与真实连续动作，teacher forcing 预测下一 patch；可选 decoder 只做解释。推理以目标图像 patch MSE 为 cost，CEM-MPC 搜动作。
4. **关键证据**：PushT 成功率 0.90（DreamerV3 0.30、IRIS 0.32）；Reach 0.92；Rope/Granular Chamfer 0.41/0.26；DINO patch 显著胜 CLS/R3M/ResNet。PushT 数据 200→18,500 时 SR 0.08→0.92；无 causal mask 在 history=3 时 0.08，有 mask 0.92；decoder loss 使 SR 0.92→0.80。MPC 明显胜 open-loop CEM/GD。
5. **确证价值**：本卷对 P1、P2、R2/R3/R4 最直接的主链证据：冻结通用感知基底 + 独立动态模型 + test-time planner。
6. **反证价值**：反证像素重建和 reward head 是通用规划必要条件；也反证 global CLS 足以支持精细接触动力学。
7. **局部可借算法**：数字蚂蚁首选主 baseline：冻结 patch feature、action-conditioned predictor、CEM-MPC、独立 probe decoder；matched 比较 pixel、learned latent、frozen global 与 frozen patch。
8. **不可外推边界**：需真实动作标签和充分覆盖；目标必须是图像；CEM 规划约 53 秒；多为模拟，未证明复杂实景、长期历史或在线适应。
9. **成熟度与裁决**：**A（主链证据）**。

### 2.2 Navigation World Models（2412.03572）

1. **论文事实**：Amir Bar、Gaoyue Zhou、Danny Tran、Trevor Darrell、Yann LeCun；Meta FAIR、NYU、BAIR；2024/2025。1B CDiT；SCAND 8.7h、TartanDrive 5h、RECON 40h、HuRoN 75h，另用 Ego4D 908h。
2. **核心问题**：大规模 action-conditioned video diffusion 是否能成为导航模拟器，以 test-time compute 做规划或轨迹重排。
3. **机制拆解**：VAE latent video；输入过去帧与 3-DoF 平面位姿增量 `(u,φ)` 加 time-shift；CDiT 交叉注意力对 context 线性扩展；推理生成像素视频，以终帧 LPIPS 和约束做 CEM，或重排 NoMaD 轨迹。
4. **关键证据**：RECON 16 秒 FVD 200.97，DIAMOND 762.73；standalone NWM ATE/RPE 1.13/0.35，优于 NoMaD 1.93/0.52；action+time、4 context、4 goals 均有消融增益。额外 Ego4D 改善 OOD Go Stanford，却损伤多数 in-domain 集。未知环境发生 mode collapse 和路径 hallucination。
5. **确证价值**：支持 pixel/video world model 作为高成本上界与 trajectory ranker；中等支持 P1/P2，不支持 R2，因为模型本身是重训练生成器。
6. **反证价值**：视频更逼真仍可能不物理；无标签视频扩大视觉先验不保证 in-domain 控制收益；“想象未知环境”不是地图证据。
7. **局部可借算法**：数字蚂蚁保留为 pixel imagination 上界；必须分别报告视频指标、action sensitivity、closed-loop goal reach 与 OOD collapse。
8. **不可外推边界**：主要 3-DoF 导航；训练昂贵；原始扩散推理慢，实时依赖 distillation/quantization；动态行人与 6-DoF 未解决。
9. **成熟度与裁决**：**A（强 world-model 上界）**。

### 2.3 LAPA（2410.11758）

1. **论文事实**：Seonghyeon Ye、Joel Jang 等；KAIST、UW、Microsoft Research、NVIDIA、AI2；ICLR 2025。Language Table 181k/442k、Bridge 60k、Open-X 970k、Something-Something-v2 200k；仿真与 3 类实机任务。
2. **核心问题**：无动作视频能否先学离散行为变化，再以少量机器人标注映射为可执行动作。
3. **机制拆解**：VQ-VAE inverse encoder 从 `(x_t,x_{t+H})` 得离散 latent action，decoder 重建未来帧；VLM 从当前图像+语言预测 latent；下游丢弃 latent head，以少量真实 7-DoF delta action 重训 action head。训练写 LM 与 action head，vision encoder 冻结；推理直接 VLA 控制。
4. **关键证据**：实机 Open-X 预训练 LAPA 50.1%，OpenVLA 43.9%；human-video LAPA 34.0%，胜 Bridge OpenVLA 30.8%；SIMPLER human-video 52.1%。Language Table 跨环境虽胜无标签基线，仍远低于 ActionVLA。latent 能聚类 2D 动作，但也编码相机运动；闭环“world model”仅定性生成。
5. **确证价值**：支持 P2/P3 与 R3/R4：行为原语可从视觉 delta 涌现，且共享 latent 可缓和 action-space 过拟合。
6. **反证价值**：细粒度抓取仍弱；“跨 embodiment latent”证据混合了语言、VLM 先验和下游重训，不能证明代码本体可执行。
7. **局部可借算法**：作为数字蚂蚁 unsupervised latent-action 正例；必须配真实 action probe、camera/distractor probe 与小标注映射。
8. **不可外推边界**：大多静态操作视频；VQ 对超参敏感；无复杂动态导航；未检验 action-correlated distractor。
9. **成熟度与裁决**：**A（潜在动作正例）**，必须与 LAOM 成对引用。

### 2.4 LAOM / Distractor Study（2502.00379）

1. **论文事实**：Alexander Nikulin 等；AIRI、MIPT、Skoltech 等；ICML 2025。DCS 4 任务，各 5,000×1,000 transitions、3 seeds；动态背景、相机抖动、颜色变化。
2. **核心问题**：无监督 latent action 在 action-correlated distractor 下是否仍会恢复可控动作。
3. **机制拆解**：LAOM 相对 LAPO 去量化、改为 multi-step IDM（最多 10 步）、共享 encoder、latent temporal consistency、增强与大 latent；监督版复用最终 decoder 所需的至多 2.5% 真动作标签，在 latent 学习期加入线性 action loss。
4. **关键证据**：架构修改使 action probe 提升约 8×、下游约 2×，仍与无 distractor 有大差距；2.5% 标签使 downstream 平均约 4.3×、归一回报 0.44。跨 embodiment 时监督版仍仅约等于少标签 BC；重建 probe 显示 LAOM 保留大量背景，multi-step IDM 才接近 control-endogenous state。
5. **确证价值**：对 P2/P3、R3/R4 是关键反证与门控证据：latent action 必须被可控性与最小性证伪。
6. **反证价值**：直接推翻“视觉变化 bottleneck 会自然抽取动作”“线性 probe 好就足够”“量化必然有利”。
7. **局部可借算法**：数字蚂蚁必须实现 distractor intervention、跨背景 action decoding、multi-step IDM、标签比例曲线和 closed-loop return；latent promotion 必须过这些门。
8. **不可外推边界**：distractor 来自 DCS，不是开放互联网；超参用真实 action probe 调优；高维 latent 的 8× probe 改善部分来自容纳全部动力学。
9. **成熟度与裁决**：**A（关键反证）**。

---

## 3. 跨形态 world model 与 generalist policy

### 3.1 HMA（2502.04296）

1. **论文事实**：Lirui Wang、Kevin Zhao、Chaoqi Liu、Xinlei Chen；MIT、UIUC、Meta FAIR；2025 预印本。3M trajectories、2.5B frames、40 datasets/embodiments，2–28 DoF；模型 3M–400M（另报约 741M XL）。
2. **核心问题**：如何在多身体、动作频率与动作维度异构下训练实时 action-video dynamics，而非单任务视频生成器。
3. **机制拆解**：身体专属 action stem/head + shared spatiotemporal trunk；2Hz、12 帧统一窗口；masked autoregression 生成 VQ 或 soft video token，动作以 MSE/diffusion；可切换 forward/full dynamics 与 policy 写面。
4. **关键证据**：HMA 比 IRASim 约 15× 快；Language Table PSNR 28.19 vs 25.41，FVD 111.52 vs 152.20；Robomimic learned evaluator 与真 simulator 的 policy ranking Pearson 0.95；10 真+90 合成轨迹达到 100 真轨迹的 100% 成功。动作条件与 modulation 消融最好；可 rollout >100 帧。
5. **确证价值**：支持 R2/R8 风格的 shared trunk + embodiment-local interface；是 cross-embodiment video WM 上界。
6. **反证价值**：full dynamics 未胜纯 forward dynamics；policy head 表现有限；高画质 evaluator 仍需人工判 success。
7. **局部可借算法**：数字蚂蚁可借 action modulation、counterfactual action sensitivity `ΔPSNR` 与 synthetic-data matched arm。
8. **不可外推边界**：所谓实时依赖 token/objective 选择；大规模结果多为生成指标；缺少真实长时 MPC；统一到 2Hz 可能抹去身体特有快动力学。
9. **成熟度与裁决**：**A（跨身体 world-model 上界）**。

### 3.2 CrossFormer（2408.11812）

1. **论文事实**：Ria Doshi、Homer Walke、Oier Mees、Sudeep Dasari、Sergey Levine；UC Berkeley、CMU；CoRL 2024。900k 轨迹、20 embodiments；130M 参数；116 次实机任务评估，另 Go1 25 分钟。
2. **核心问题**：不手工对齐观测与动作空间，单一策略能否同时覆盖 manipulation、navigation、locomotion、aviation。
3. **机制拆解**：按模态 token 化多相机/proprio；shared causal transformer；4 类 action head（7D 单臂、2D waypoint、14D 双臂、12D 四足），不同 chunk/frequency；语言或 goal image 条件。
4. **关键证据**：平均 73%，单机器人同架构 68%，各域最佳 prior 51%；对手工对齐方法约 3×；未见 Tello 身体复用 navigation head 达 0.82 路径进度。论文明确承认没有显著跨身体正迁移。
5. **确证价值**：支持 P2、R2/R8：共享表征可吸收异构数据，但 owner-local action interface 仍不可省。
6. **反证价值**：反证必须手工统一动作空间；同时反证“多身体训练自然产生大正迁移”。
7. **局部可借算法**：数字蚂蚁 cross-substrate transfer 应冻结 shared trunk，只重训 sensor stem/action head，并与 from-scratch、手工对齐、共享 head 对照。
8. **不可外推边界**：采样权重手调；目标数据被上采样；多个任务 trial 数小；非 world model，无未来状态或反事实预测。
9. **成熟度与裁决**：**A（cross-embodiment policy 主基线）**。

### 3.3 HPT（2409.20537）

1. **论文事实**：Lirui Wang、Xinlei Chen、Jialiang Zhao、Kaiming He；MIT、Meta FAIR；NeurIPS 2024。52 datasets、约200k–270k trajectories；trunk 3.1M–1.1B；仿真 50 episodes×5 runs，实机每任务约100 demos、15 trials。
2. **核心问题**：视觉与 proprio heterogeneity 能否通过身体专属 tokenizer 对齐到共享 policy representation，再迁移至新身体。
3. **机制拆解**：stem 把视觉/proprio 压为固定 token；shared transformer trunk；task/embodiment head 输出动作。预训练监督 BC；迁移重建 stem/head，冻结或微调 trunk。
4. **关键证据**：异构数据、模型与 compute 的 validation loss 有规模趋势；跨仿真/实机迁移较 scratch 提升，Sweep Leftover HPT-XL 76.7%、scratch 43.3%；去 vision 或 proprio 均变差。闭环成功通常仍低于 90%。
5. **确证价值**：支持 P1/P2 与 R2，尤其是稳定共享基底和身体接口分离；但论文主结果并非运行时冻结证据。
6. **反证价值**：validation loss 与闭环任务之间存在两层 evaluation gap；共享 trunk 的收益依赖数据配比与后训练。
7. **局部可借算法**：作为数字蚂蚁 frozen/shared trunk 迁移 baseline；记录 frozen、full-finetune 与 scratch 三臂。
8. **不可外推边界**：预训练是 action-supervised BC；人类视频使用手部 proxy action；任务多为短时固定身体；数据混合与清洗未系统解决。
9. **成熟度与裁决**：**A（共享 substrate 迁移证据）**。

### 3.4 Octo（2405.12213）

1. **论文事实**：Octo Model Team；UC Berkeley、Stanford、CMU、Google DeepMind；RSS 2024。800k episodes、25 OXE datasets；27M/93M；9 实机平台；fine-tune 约100 demos、每设置20 trials。
2. **核心问题**：开放、模块化 generalist policy 能否零样本控制多机器人，并快速适配新传感器和动作空间。
3. **机制拆解**：语言/goal/image tokenizers + block-wise transformer + readout token；小 diffusion head 生成连续 action chunk；fine-tune 可新增 observation tokenizer/action head。
4. **关键证据**：零样本平均比 RT-1-X 高 29%；六个新域 fine-tune 平均 72%，scratch 20%、VC-1 15%。diffusion head 83%，MSE 35%，离散动作 18%；25 数据集胜 11 数据集与单机器人；新 skill 零样本仅 5%。
5. **确证价值**：支持跨身体接口模块化与连续多模态 action head；是 flat generalist-policy 标准对照。
6. **反证价值**：大规模多任务并不产生未见 skill；proprio 甚至因 causal confusion 变差；wrist camera 与语言受数据缺口限制。
7. **局部可借算法**：数字蚂蚁保留 Octo-style flat policy，和 world-model planner、hierarchy、latent action 做 matched 对照。
8. **不可外推边界**：仅 imitation、主要单/双臂操作；不是 world model；无在线学习；数据以成功示范为主。
9. **成熟度与裁决**：**A（flat policy 强基线）**。

### 3.5 FAST（2501.09747）

1. **论文事实**：Karl Pertsch、Kyle Stachowicz 等；Physical Intelligence、UC Berkeley、Stanford；RSS 2025。FAST+ 用 1M action chunks；π0-FAST 训练约 10k 小时/903M timesteps；6 实机+1 仿真任务。
2. **核心问题**：高频连续动作逐维逐时刻离散造成 token 高相关与“复制上一步”捷径，如何压缩后再自回归。
3. **机制拆解**：动作按 1/99 分位归一；逐维 DCT；系数量化；低频优先展平；BPE 压缩。它是 action tokenizer，不是 latent semantic option 或 world model。
4. **关键证据**：50Hz 双臂 700 token→约53；naive 在20/50Hz任务几乎失败，FAST/FAST+成功；π0-FAST 与 diffusion π0 相当，训练 GPU 时约少5×；去 BPE 变差。推理约750ms/chunk，diffusion π0 约100ms。
5. **确证价值**：支持高频 action interface 需要压缩；对 P2 仅工程性支持，不支持 P3“涌现 switching”。
6. **反证价值**：反证 action token 数越细越精确；也反证训练快必然推理快。
7. **局部可借算法**：仅在数字蚂蚁高频 actuator baseline 中使用；与 raw continuous、naive bins、learned latent action 分开报告。
8. **不可外推边界**：主要静态 manipulation；频域压缩不提供因果、语义或跨身体可执行对应；动态任务受推理延迟限制。
9. **成熟度与裁决**：**B（强动作编码基线）**。

---

## 4. VLA、语言层级、数据生成与空间接口

### 4.1 GR00T N1（2503.14734）

1. **论文事实**：NVIDIA 团队，2025 技术报告；公开 2.2B 模型。真实 GR-1 数据 88h；视频模型扩至 827h；DexMimicGen 780k trajectories/约6500h；预训练约50k H100 GPU-hours。
2. **核心问题**：人形机器人如何把 web/human、simulation、neural video 与真实动作数据组合为 generalist VLA。
3. **机制拆解**：Eagle-2 VLM“System 2”约10Hz；embodiment-specific state/action MLP + DiT“System 1”约120Hz，16-step chunk、4 次 flow denoise；两部分紧耦合训练，语言层冻结、视觉与动作层写入。
4. **关键证据**：仿真多身体与 GR-1 实机优于多种 imitation baseline；latent action 检索跨人体/机器人呈相似方向；中层 VLM feature 优于末层；neural trajectory 与 simulation 数据有数据效率收益。报告规模大，但多项精确比较依赖内部数据。
5. **确证价值**：支持 cross-embodiment local adapter、低/高频分层与数据金字塔；支持 P1/P2。
6. **反证价值**：“System 1/2”命名不等于 R2 owner 分离：VLM 与 DiT 端到端紧耦合，且 latent action 仍暴露 distractor 风险。
7. **局部可借算法**：作为大规模 VLA 上界；借身体专属 encoder/decoder、真实/仿真/生成数据分层审计，不借整体联合训练。
8. **不可外推边界**：高计算、内部数据、部分闭源生成/筛选器；视频伪动作可能复合误差；人形操作远离数字蚂蚁资源约束。
9. **成熟度与裁决**：**B（工业上界 / 边界反例）**。

### 4.2 Gemini Robotics（2503.20020）

1. **论文事实**：Google DeepMind Gemini Robotics Team，2025 技术报告，64 页；基于 Gemini 2.0。ERQA 400 题；多平台真实操作；新短任务可用约100 demos 适配。
2. **核心问题**：frontier VLM 的 embodied reasoning 能否连接高频、灵巧、开放词汇机器人控制，并适配新身体。
3. **机制拆解**：Gemini Robotics-ER 输出检测、点、轨迹、抓取、多视图对应与3D框；Gemini Robotics VLA 直接输出反应式动作；另有 specialization 到双臂与高 DoF 人形。训练/推理细节与完整数据规模未开放。
4. **关键证据**：ERQA Gemini 2.0 Pro 48.3%，CoT 54.8%；Robotics-ER 在 Paco-LVIS/Pixmo-Point/Where2Place 为71.3/49.5/45.0，SUN-RGBD AP@15 48.3；报告展示开放词汇、长时折纸/纸牌等实机能力与少样本适配。
5. **确证价值**：显示 VLA/ER 的工业能力上界；对空间 affordance、语言约束与跨身体提供行为证据。
6. **反证价值**：QA、pointing 与轨迹图不是闭环控制等价物；CoT 增益不能证明内部因果世界模型。
7. **局部可借算法**：仅作闭源 VLA oracle 与 spatial waypoint 上界；数字蚂蚁不能把其输出当 ground truth，应通过可执行闭环验证。
8. **不可外推边界**：闭源模型、数据、训练和大量评估细节；难以复现；安全结论是工程流程而非形式保证。
9. **成熟度与裁决**：**B（闭源能力上界）**。

### 4.3 Hi Robot（2502.19417）

1. **论文事实**：Lucy Xiaoyang Shi 等，Physical Intelligence、Stanford、UC Berkeley；ICML 2025。3 平台；table bussing、sandwich、shopping；每任务/方法20 trials；PaliGemma-3B + π0。
2. **核心问题**：复杂提示、中途反馈和用户约束能否通过语言高层分解为 VLA 可执行的原子任务。
3. **机制拆解**：高层 VLM 约1秒或新反馈时输出语言子任务/口头回复；低层 π0 flow policy 输出连续 chunk；用真实 skill 标签与 VLM 反向生成的合成交互训练高层。
4. **关键证据**：Hi Robot instruction accuracy 平均比 GPT-4o 高逾40点；去合成数据平均 IA/TP 约降46/39点；同数据 flat VLA 比 hierarchy 低约34/19点；人类高层 oracle 显示主要瓶颈在 reasoning。
5. **确证价值**：支持多时间尺度接口和 language waypoint 强 baseline；对 P3 只证明显式语言层级，不证明 emergent latent switching。
6. **反证价值**：反证 flat VLA 足以处理开放反馈；但也显示高层缺 memory、近物体偏置和 OOD recovery。
7. **局部可借算法**：数字蚂蚁将其作为“显式语言子目标”对照，与非语言 `z_t/β_t` 在同任务、同低层控制器下比较。
8. **不可外推边界**：高层按任务分别训练、合成 prompt 依赖工程；语言不是低带宽昆虫控制接口；无长期内部状态。
9. **成熟度与裁决**：**B（语言层级强基线）**。

### 4.4 NaVILA（2412.04453）

1. **论文事实**：An-Chieh Cheng 等；UCSD、USC、NVIDIA；2024/2025。2k YouTube touring videos→20k trajectories；R2R/RxR、1,077 条 VLN-CE-Isaac；实机25指令×3次；8B VLA。
2. **核心问题**：语言导航如何跨越低频语义规划与高频腿部控制，不把 joint action 强塞给 VLM。
3. **机制拆解**：VLA 从历史+当前 RGB 输出“前进75cm/右转30°”；解析成速度与时长；LiDAR height-map PPO policy 输出12关节位置。高层约1FPS，低层实时。
4. **关键证据**：R2R-CE SR 54%、NaVid 37%；human video 消融使 SR 49.7→54.0；Isaac Go2 vision 50.2%、blind 36.2%；低层 collision 0.81、ROA 3.09；实机总体约88%，复杂指令约75%。
5. **确证价值**：支持高低频 owner 与 language waypoint interface；跨 Go2/H1/T1 复用高层是 cross-substrate 正证据。
6. **反证价值**：VLA 不会障碍规避时，低层 sensing 仍必须拥有安全控制；语言 waypoint 不是完整动作模型。
7. **局部可借算法**：数字蚂蚁保留 `language waypoint + local reactive controller` 强 baseline，并强制与 latent waypoint 同 sensor/actuator 对照。
8. **不可外推边界**：正则解析语言动作是协议匹配，不是开放控制；高层仅少数动作类型；依赖 LiDAR、pose estimation 与大量仿真。
9. **成熟度与裁决**：**A（层级导航主基线）**。

### 4.5 DreamGen（2505.12705）

1. **论文事实**：Joel Jang 等；NVIDIA、UW、KAIST 等；2025 预印本。RoboCasa 最多240k neural trajectories；9 实机任务、3 身体；GR1 2,884 条真轨迹，14 新行为+13 新环境任务；视频生成一次大实验用1500 L40×54h。
2. **核心问题**：video world model 能否作为 rare-heavy 数据生成器，而非实时 planner，扩展行为与环境覆盖。
3. **机制拆解**：LoRA 微调 image-to-video；语言+初帧生成视频；IDM 或 LAPA 恢复伪动作；以1:1混合真/合成或只用合成训练 DP/π0/GR00T N1。
4. **关键证据**：RoboCasa synthetic scale 呈 log-linear 增益，only-neural 达20.55%；GR1/Franka/SO-100 均改善；新行为 11.2→43.2%，新环境0→28.5%。DreamGen Bench 的 instruction/physics 分数与下游表现正相关。
5. **确证价值**：支持 rare-heavy data augmentation 与生成器独立评估；与 R2/R15 的离线 artifact refresh 更相容，而非在线写 substrate。
6. **反证价值**：生成视频 + IDM/LAPA 会形成 hallucination×inverse-error 复合链；自动 physics judge 本身会 hallucinate。
7. **局部可借算法**：数字蚂蚁仅设离线 synthetic arm；保存生成模型、prompt、筛选、伪动作 provenance；真环境闭环是最终门。
8. **不可外推边界**：算力极高、手工初帧、任务仍简单；没有证明复杂长时新技能；未与所有人类视频方法直接比较。
9. **成熟度与裁决**：**B（rare-heavy 数据生成候选）**。

### 4.6 SpatialVLA（2501.15830）

1. **论文事实**：Delin Qu 等；上海 AI 实验室、复旦、上交、浙大等；2025 预印本。PaliGemma2；1.1M 实机 episodes；24 实机任务、3 仿真环境。
2. **核心问题**：跨机器人 VLA 如何对齐相机视角下的3D观察与连续空间动作。
3. **机制拆解**：ZoeDepth+相机内参反投影到 egocentric 3D；SigLIP feature 加3D sinusoidal embedding；7D 动作变成 translation/rotation/gripper 3 token，自适应 Gaussian quantile grids；新身体重离散并插值 action embedding。
4. **关键证据**：大范围 simulation/real zero-shot 与 post-training 优于 OpenVLA、RT-X、Octo 等；仅3 token/action，报告约20.1Hz；Ego3D 与 adaptive grid 有消融收益。
5. **确证价值**：支持空间 reference frame 与 body-local action adapter；为数字蚂蚁测 egocentric 3D 提供强接口 baseline。
6. **反证价值**：所谓 robot-agnostic 仍依赖外部单目深度、内参和每域 action normalization；离散网格不是语义 latent action。
7. **局部可借算法**：加入 ego/world/object frame 反事实 probe；action grid 仅作离散空间动作 baseline。
8. **不可外推边界**：深度误差会直接污染控制；主要是 in-distribution 与 post-train 适配；未证明 distractor robustness、长期世界模型或跨 substrate 零样本。
9. **成熟度与裁决**：**B（空间动作接口强基线）**。

### 4.7 RoboSpatial（2411.16537）

1. **论文事实**：Chan Hee Song 等；Ohio State、NVIDIA；2024–2026。1M images、5k 3D scans、3M spatial QA；训练实际均衡采样900k；200+ 实机 query。
2. **核心问题**：VLM 缺少 ego/world/object reference frame 与可放置空间、兼容性数据，能否通过几何生成数据补足。
3. **机制拆解**：由3D bbox、相机内外参与 occupancy 生成 configuration/context/compatibility QA；2D/3D VLM instruction tuning；实机输出点，经 SAM2、depth 与 cuRobo 执行。
4. **关键证据**：RoboPoint 总分38.9→70.6，LLaVA-NeXT 30.3→60.5；实机 LLaVA-NeXT 23.7→52.6，GPT-4o 46.9；数据100k→3M 时30.3→72.4。辅助 grounding 单独仅32.4，空间数据51.8，联合60.5。
5. **确证价值**：强支持 reference-frame probe 与空间评估资产；对控制只给模块化闭环证据。
6. **反证价值**：空间 QA 好不等于动态控制；2像素可对应5–10cm实机误差；完整3D扫描模型不一定适合在线部分观测。
7. **局部可借算法**：把 ego/world/object frame、free-space compatibility、stack/occlusion 变成数字蚂蚁只读 eval。
8. **不可外推边界**：静态室内/桌面、无人与动物；top-down occupancy 不支持容器/下方空间；运动规划由外部系统完成。
9. **成熟度与裁决**：**B（评估资产，不是 world-model 主链）**。

---

## 5. 数字蚂蚁 matched baseline 矩阵

所有 baseline 必须共享：相同传感输入、动作频率、训练轨迹、环境步预算、随机种子、planner 候选数、低层安全 controller、成功判据与 wall-clock 上限。生成方法另报告训练与推理能耗。

| 轴 | B0 | B1 | B2 | B3 | 必报指标 |
|---|---|---|---|---|---|
| 状态表征 | raw proprio | frozen global feature | frozen patch feature（DINO-WM） | learned next-embedding（NE-Dreamer） | closed-loop SR、OOD SR、feature drift、线性/非线性 action probe |
| 世界模型写面 | 无模型 flat policy | reward-centric latent | decoder-free latent | pixel/video diffusion | one/multi-step error、action sensitivity、MPC SR、延迟、显存 |
| 规划 | reactive BC | open-loop CEM | receding-horizon CEM/MPC | actor on imagined latent | SR、路径效率、replan 次数、model exploitation |
| 动作接口 | raw continuous | naive bins / FAST | supervised body-local adapter | unsupervised latent + decoder | action MSE、闭环 return、频率鲁棒性、解码标签效率 |
| 层级 | flat policy | language waypoint（NaVILA） | learned latent waypoint | learned latent + support constraint | horizon、subgoal OOD率、恢复率、低层拒绝率 |
| 记忆 | 当前帧 | dense short video | text/structured long memory | short video + structured event snapshot | 遮挡恢复、计数、长时任务、摘要漂移、失败保留率 |
| 跨 substrate | scratch | shared trunk frozen | shared trunk full-finetune | shared trunk + new stem/head | label efficiency、正/负迁移、旧身体回归、适配参数量 |
| 数据生成 | 真数据 only | render/domain randomization | video+IDM | video+latent action | 真环境 SR、合成比例曲线、physics violation、provenance |
| 空间接口 | 2D RGB | ego-depth | language waypoint | ego/world/object frame explicit | frame swap accuracy、point-to-3D误差、碰撞、可放置性 |

推荐最小主实验是四臂：`flat BC`、`DINO-WM+CEM-MPC`、`NE-Dreamer`、`language waypoint+reactive low-level`。latent action、pixel world model、synthetic data 与 cross-substrate transfer 作为后续扩展，不应先于这四个 matched baseline。

## 6. Latent action 可控性 probes

### 6.1 必做 probes

1. **真实动作可解码性**：冻结 latent，报告线性与小 MLP 对真实动作的 MSE/R²；只作必要条件，不作充分条件。
2. **反事实动作敏感性**：固定 `o_t`，交换动作/latent，未来状态必须按可执行方向改变；报告 `Δprediction` 与真实 counterfactual 一致率。
3. **distractor invariance**：固定动力学，替换背景、纹理、相机抖动、旁观蚂蚁；latent action 应稳定，state latent 可变化。
4. **camera-motion leakage**：仅移动相机不动身体；若 latent action 强变化则判污染。
5. **minimality**：由 latent 重建 distractor 的能力应受限，同时保留控制充分性；高维“什么都编码”不能过门。
6. **multi-step controllability**：单步 probe 通过后，检查5/10/20步 rollout 的动作序列可执行性、误差增长与返回。
7. **code intervention**：直接替换 latent code，观察动作方向、速度、转向是否单调且跨初态一致。
8. **label-efficiency curve**：0、0.5%、1%、2.5%、5%、10% 真动作监督；与同标签预算 BC、IDM 比较。
9. **cross-body semantics**：同一 code 在不同 substrate 上不要求同 joint 值，但要求同任务空间效果；报告 effect-space 而非 token-space 对齐。
10. **closed-loop utility**：最终必须提升真环境成功率/回报，且不是只降低 imitation loss 或 prediction loss。

### 6.2 Promotion gate

latent action 只有同时满足以下条件才能从观察项升为 shadow controller：真实动作可解码；distractor/camera leakage 低于阈值；counterfactual effect 正确；closed-loop 胜同标签预算 BC/IDM；跨 seed 稳定；失败时可退回 raw/body-local action。任何一项缺失都不得宣称“涌现了行为原语”。

## 7. Cross-substrate transfer 设计

1. **定义 substrate**：至少包括传感布局、动作维度、频率、动力学与身体尺度中的两项变化；只换纹理不算跨 substrate。
2. **训练协议**：源身体训练 shared trunk；目标身体仅提供固定预算（如 10/50/100/500 trajectories）；比较 scratch、frozen trunk+new stem/head、full fine-tune、manual action alignment、shared universal head。
3. **测量效应空间**：将不同身体动作映射到共同 task effect（位移、转角、搬运结果、巢向进展），禁止用 raw joint/token 相似度冒充迁移。
4. **保留旧身体回归**：适配目标后重测源身体；full fine-tune 的正迁移若伴随旧身体遗忘，不满足 R2/R15。
5. **接口所有权**：sensor stem、action decoder 与安全低层由目标 substrate owner 管理；shared trunk 只发布不可变 feature/state snapshot。
6. **可回滚**：目标 adapter 独立版本化；达到退出条件前只在 SHADOW；失败恢复到 scratch/body-local controller。

## 8. Kill conditions

1. **冻结特征失效**：在 matched 预算下，frozen patch world model 的闭环成功率持续低于 raw-state/learned-latent baseline，且误差集中于任务必要变量，kill “frozen feature 足够”。
2. **像素幻觉**：FVD/LPIPS 改善但 action sensitivity、MPC 成功或物理一致性不升，kill “视频质量代表 world-model quality”。
3. **latent distractor leakage**：背景/相机干预使 action code 变化达到真实动作变化同量级，或新 distractor 下真实动作解码崩溃，kill latent-action promotion。
4. **latent 无闭环价值**：probe 好但 closed-loop 不胜同标签预算 BC/IDM，kill “可解码即有用”。
5. **跨身体无正迁移**：frozen shared trunk 在多个目标身体均不胜 scratch，或只靠目标数据上采样取胜，kill cross-substrate claim。
6. **接口塌缩**：共享 universal action head 导致任一身体显著回归，而 body-local head 可恢复，kill universal action-space 方案。
7. **语言层级 OOD**：language waypoint 频繁超出低层 support，且拒绝/恢复机制不能降低失败，kill hierarchy promotion。
8. **在线更新越界**：RTRRL 类更新造成 drift、静止策略、源任务遗忘或无法回滚，立即停更并退回 frozen controller。
9. **合成误差复合**：video hallucination 与 IDM/latent decoder 错误在真环境累积，使 synthetic arm 低于真数据-only，kill 该生成器/伪动作组合。
10. **记忆摘要漂移**：语言/结构化 long memory 丢失失败、约束或 provenance，导致重复错误或错误完成声明，kill summary writer。
11. **空间 QA 不闭环**：reference-frame 分数提高而碰撞、放置或导航无改善，禁止将 QA 模型提升为 controller。
12. **评估反灌**：若 benchmark/自动 judge 被直接作为在线 reward 并出现 Goodhart，立即停止；evaluation 保持只读。

## 9. 最终行动裁决

- **立即进入 benchmark**：DINO-WM、NE-Dreamer、LAOM distractor suite、NaVILA language waypoint、Octo flat policy。
- **进入 shadow / 扩展实验**：HMA、HPT/CrossFormer transfer、MEM 双时间尺度、SpatialVLA/RoboSpatial reference-frame probes。
- **仅 rare-heavy**：DreamGen synthetic generation、GR00T N1 式数据金字塔。
- **仅作上界或反例**：Navigation World Model 的像素生成、Gemini Robotics 闭源 oracle、RTRRL 无安全门在线全策略更新、FAST token action。
- **不得直接宣称**：视频生成器是物理 world model；latent code 是可控 option；跨身体共享 trunk 是通用动作空间；空间 QA 是闭环控制；“System 1/2”命名等于 VZ 的 frozen substrate / adaptive controller 边界。
