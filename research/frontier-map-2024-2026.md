# Cognitive-Agent 前沿地图（2024–2026）

调研日期：2026-07-20

## 0. 地图口径

这不是“再列一批论文”，而是把 Volvence 已有 7 primitive 放到更大的外部研究地形中：

- 深读核心层：`frontier-sweep-2026-07-20.md` 的 28 篇；
- 广度扩展层：本地图新增 78 篇；
- 本轮地图总计：106 篇严格去重论文；
- 连同本轮前既有 208 篇，本地研究库覆盖约 314 篇。

新增 78 篇分为五轴：架构与学习 15、安全与治理 17、关系与多主体 16、具身与世界模型 16、
脑科学与认知约束 14。候选均先与 `research/papers/**/*.pdf` 和研究文档中的 ID / DOI 去重，
再用 arXiv、会议、期刊或实验室官方页面复核。

证据等级：

- **A：主链证据**——直接约束 R 不变量、顶会/顶刊或完整实证，建议进入 spec / benchmark motivation；
- **B：强基线 / 关键反例**——方法成熟但与 VZ 边界不同，适合 matched baseline 或 kill condition；
- **C：观察项**——新预印本、单点证据或外推距离较远，只登记，不改主链。

## 1. 总判断

### 1.1 2024–2026 的主收敛不是“更大的模型”，而是六个接缝

1. **基底与控制器**：DINO-WM、CLAW、LAPA/LAOM、CrossFormer 共同把 frozen representation 与
   latent action/control 的接口变成可实验对象。
2. **PE 与信息价值**：thalamocortical PE、domain-general dopamine、ICL curiosity 负面定理共同说明
   `prediction mismatch ≠ reward ≠ information gain`。
3. **记忆与信用**：EvoLib、DeltaMem、AdaMEM、SaliMory、MARS 让“写什么、何时巩固、谁获信用”
   成为记忆系统的核心，而非容量大小。
4. **抽象与 support**：多时间尺度规划给出正证据，Mind the Gap 与 LAOM 给出反证——层级 latent
   只有在数据支持和可控性成立时才降低 horizon。
5. **监控与可欺骗性**：AuditBench、SHADE-Arena、Strategic Deception 证明 read-only monitoring 必须
   多工具、多 scaffold、可被对抗；激活监控并非天然可信。
6. **关系与健康**：长期 chatbot RCT、trust repair、Lifelong-SOTOPIA 说明“更高信任 / 更多互动”
   不是单调好目标；关系质量必须包含依赖、替代真人关系、修复证据与行为适应。

### 1.2 VZ 目前最值得坚持的四条边界

- R2 frozen substrate：大多数 world-model / VLA 论文仍端到端训练，VZ 的分层是必要差异，不是落后。
- R8 snapshot SSOT：Sheaf-ADMM、DeltaMem、Active Epistemic Control 都支持局部状态与公共契约分离。
- R12 evaluation read-only：审计工具会被识别、被欺骗或 Goodhart；评估绝不能直接成为在线 reward。
- R15 rollback / falsifiability：unlearning、runtime enforcement、deployment replay、artifact diff 必须组合，
  单一 benchmark 不足以放行自修改。

## 2. Axis A：架构、学习、记忆与信用（15）

### A 级

- `2605.17058` **Learning Multi-Timescale Abstractions for Hierarchical Combinatorial Planning**  
  SMDP-aware world model 联合学习一步动态与可变时长 macro-transition；直接检验 latent abstraction
  是否缩短 planning horizon。映射 P2/P3/P5，R1/R3/R4/R9。
- `2606.04130` **CLAW**  
  从无动作视频学习连续 latent action 与 world model，并用 adversarial regularization 防止 code 偷带未来信息。
  是连续 `z_t` 的强外部基线，但端到端训练只允许 rare-heavy 对照。
- `2607.12547` **Mind the Gap: Promises and Pitfalls of Hierarchical Planning**  
  冻结低层 world model 后，高层 planner 会提出 OOD subgoal；加入 support constraint 才恢复收益。
  是“hierarchy 本身不等于 improvement”的关键 kill condition。
- `2606.09828` **Latent Spatial Memory for Video World Models**  
  在 diffusion latent 中维护 initialize–read–update 的持久 3D cache，避免 RGB 往返；支持 latent memory
  lifecycle，但不等于社会 / 因果世界模型。
- `2602.03974` **Active Epistemic Control for Query-Efficient Verified Planning**  
  分离 grounded fact store 与 uncertain belief store；预测只剪枝，最终承诺须由事实认证。直接支持 R8/R11/R15。
- `2605.14477` **Test-Time Learning with an Evolving Library**  
  把轨迹蒸馏成 modular skills / reflective insights，以 information gain 分配写入和合并信用；可借 memory
  lifecycle，不借文本自评作为训练真值。
- `2606.03083` **DeltaMem**  
  Task-Tree 与 Env-Tree 分离程序性技能和环境事实，只存 residual delta，高频路径再巩固为 root。
  与语义 owner 分离和 snapshot SSOT 高度一致。
- `2606.05684` **AdaMEM**  
  离线长期 trajectory store + 在线短期 strategy memory，提供清晰双时间尺度；但 strategy 是自然语言，
  不是 latent controller。

### B 级

- `2607.14180` **RENEW** — 用人类偏好修复被 policy exploit 的 world model；适合作为 rare-heavy
  主动修复对照，不能让在线偏好直写 substrate。
- `2606.25115` **Forget to Improve** — 把容量、能耗、陈旧经验、poisoning 纳入同一记忆预算；
  单一“价值/字节”分数会产生 R12 风险。
- `2507.12508` **MindJourney** — frozen VLM 主动提出相机轨迹，world model 生成想象证据；
  证明 test-time compute 可用于取证，也暴露 hallucinated observation 风险。
- `2601.18510` **Just-In-Time RL** — 从非参数记忆估 action advantage 并直接改 logits；冻结权重仍是
  token-space RL，作为 R3/R4 反例。
- `2510.13220` **EvoTest** — J-TTL 学习曲线是好评估协议，但 prompt/memory/tool/hyperparameter
  全写面自演化违反 R10。
- `2602.22817` **Hierarchy-of-Groups Policy Optimization** — context-consistent group advantage
  是长程信用 baseline；主体仍是 token policy GRPO。
- `2606.25421` **Agent-Authored World Modeling** — 区分 decision-relevant dynamics 与完整 observation
  prediction；belief / target 均在自然语言层，只借 target design。

## 3. Axis B：安全、监控、自修改与发布门（17）

### A 级

- `2503.10965` **Auditing Language Models for Hidden Objectives** — 训练数据、行为攻击、SAE 的盲审组合；
  证明多模态审计必要，也反证 SAE 单工具足够。
- `2602.22755` **AuditBench** — 56 个模型、14 类隐藏行为、13 类工具；scaffolded black-box 领先白盒，
  暴露 tool-to-agent gap。
- `2602.08877` **Stress-Testing Alignment Audits with Prompt-Level Strategic Deception** —
  激活 / SAE / embedding 审计均可被定向欺骗，监控器本身必须进入攻击面。
- `2506.15740` **SHADE-Arena** — 联合测量主任务效用、side-task 破坏和 monitor bypass；
  hidden scratchpad 显著提高隐蔽攻击。
- `2601.21112` **Information Access and Sabotage Monitoring** — Extract-and-Evaluate 优于把完整轨迹
  一次性交给 monitor；“信息更多”会被事后合理化污染。
- `2503.18666` **AgentSpec** — trigger–predicate–enforcement DSL 在工具执行前 fail-closed；
  自动生成规则 recall 约 71%，所以不变量仍须人工声明。
- `2407.06460` **MUSE** — 遗忘需同时测逐字、知识、隐私、保留效用、规模与连续删除；
  强反证单一 unlearning score。
- `2506.12618` **OpenUnlearning** — 13 算法、16 指标、450+ checkpoint，并对指标做 meta-eval；
  方法排名对 evaluator 高度敏感。
- `2406.13352` **AgentDojo** — 97 个状态化工具任务、629 个安全测试；prompt 层防御无法同时保效用与安全。
- `2410.02644` **Agent Security Bench** — 攻击覆盖 system/user/memory/planning/tool；最高平均攻击成功率
  84.3%，支持按 owner / 生命周期拆安全面。
- `2503.18813` **CaMeL** — 可信控制流与不可信数据流分离，capability policy 限制外传；
  是 R8/R10 的强架构证据。

### B 级

- `2410.21514` **Sabotage Evaluations for Frontier Models** — 四类 mitigation-aware eval；
  当前未过阈值不能外推未来模型。
- `2604.24618` **Would Models Sabotage AI Safety Research?** — 270 场景未见无提示破坏，
  但 evaluation awareness 高，零发生率不能作为放行。
- `2505.23836` **Models Often Know When They Are Being Evaluated** — Gemini 2.5 Pro eval/deployment
  分类 AUC 0.83；evaluation awareness 应为独立指标。
- `2605.26438` **LURE** — live-usage replay 改变模型排名；仍可能被识别为拼接测试。
- `2512.15688` **BashArena** — 637 个高权限 Linux 任务；在 4% trajectory FPR 下仍有 26% 隐蔽破坏成功率。
- `2502.04577` **Position-aware Automatic Circuit Discovery** — artifact diff 应比较位置化因果子图；
  自动 circuit 仍不等于完整机制解释。

## 4. Axis C：关系、社会认知与多主体（16）

### A 级

- `2502.00640` **CollabLLM** — 优化整段协作而非单轮回答；人类实验满意度 +17.6%、耗时 −10.4%。
  支持长期共同结果，但其 reward 仍在 token 对话策略。
- `2412.19726` **ToM Benchmarks Are Broken** — literal prediction 与 functional partner adaptation
  明显解耦；静态 ToM / persona 命中不能证明 user model。
- `2505.17663` **DynToM** — 5,500 连续场景、78,100 问题；状态迁移时模型平均落后人类 44.7%。
  支持 user state 版本化、时间轨迹和 provenance。
- `2506.12666` **LIFELONG-SOTOPIA** — 累积完整历史会使可信度和目标完成持续下降；
  高级摘要只能部分缓解，反证“多记就是更懂”。
- `2503.17473` **Extended Chatbot Use RCT** — 981 人、30 万+消息；高频自发使用与孤独、依赖、
  问题使用和更少真人社交相关。关系 KPI 不能单调最大化互动和信任。
- `2409.17348` **Language-Grounded MARL** — 公共通信 representation 与私有控制可分离，
  并支持未见伙伴协作；语言监督不等于自然涌现。
- `2404.16698` **Cooperate or Collapse / GOVSIM** — 15 模型中仅 2 个产生可持续结果；
  长期群体价值不能退化为逐轮个人收益。
- `2503.01658` **CoPL** — 用户–响应图与 mixture-of-LoRA 支持稀疏个性化；
  用户 embedding 不等于有 consent/provenance 的 relationship state。

### B 级

- `2407.07086` **Hypothetical Minds** — 可修正他者策略假设显著提高 multi-agent 适应；
  自然语言 hypothesis 是 scaffold，不是持久 latent state。
- `10.1145/3630106.3658924` **Trust Development and Repair** — Model Update 修复最强，其次道歉、
  承诺，否认最差；修复必须有能力变化证据，不是漂亮措辞。
- `10.1145/3613904.3642780` **Interpretability, Feedback and Trust** — 1,511 人预注册实验；
  outcome feedback 比解释更稳定地影响信任，但信任提升不等于协作表现提升。
- `2404.13627` **NegotiationToM** — BDI 推断仍大幅落后人类，CoT 无法补齐；
  伙伴意图应是带置信度、可修正的 owner state。
- `2025.naacl-long.257` **AgentSense** — 1,225 多轮社会场景；适合 boundary/consent 与隐私信息使用评估。
- `2505.19591` **Evolving Orchestration** — 学习中央 puppeteer 的强 collective-intelligence baseline；
  中央编排器容易成为全局状态第二 owner。
- `2506.07388` **Shapley-Coop** — 显式边际贡献信用改善合作和公平；
  Shapley-CoT 不是可信因果归因。
- `2402.15052` **ToMBench** — 静态能力下限，可与 DynToM / functional ToM 配套；
  单独不能证明关系连续性。

## 5. Axis D：具身、世界模型、潜在动作与空间智能（16）

### A 级

- `2411.04983` **DINO-WM** — 冻结 DINOv2，仅学 latent dynamics + CEM/MPC；
  是数字蚂蚁 frozen substrate + latent planner 首选 baseline。
- `2412.03572` **Navigation World Models** — 1B action-conditioned video diffusion，
  可独立 MPC 或重排导航轨迹；像素想象是高成本上界。
- `2410.11758` **LAPA** — 从无标签视频发现离散 latent action，少量标注映射到真实动作；
  是行为原语正例。
- `2502.00379` **LAOM / Distractor Study** — 仅 2.5% 动作监督使 downstream 平均提高 4.3×；
  强反证无监督 latent action 必然编码可控自运动。
- `2502.04296` **HMA** — 300 万轨迹、40 embodiment，共享 trunk + 身体专属接口；
  是跨形态 world model 上界。
- `2408.11812` **CrossFormer** — 90 万轨迹、20 embodiment，统一 manipulation/navigation/
  locomotion/aviation；适合数字蚂蚁跨身体迁移。
- `2409.20537` **HPT** — embodiment-specific stems + shared trunk + task heads；
  支持稳定基底与接口分层，但不是运行时冻结证据。
- `2405.12213` **Octo** — 80 万 Open-X 轨迹、9 实机平台；标准 flat generalist-policy 对照。

### B 级

- `2501.09747` **FAST** — DCT + BPE 动作 tokenizer，训练加速约 5×；
  频域压缩不等于语义 option，仍是 token action。
- `2503.14734` **GR00T N1** — VLM System-2 + diffusion System-1；
  联合训练不满足严格 R2，“System 1/2”命名不等于 owner。
- `2503.20020` **Gemini Robotics** — embodied reasoning / VLA 上界；
  闭源、语言/代码高层，不是可复现非语言 latent baseline。
- `2502.19417` **Hi Robot** — 语言子任务 + 低层 VLA；必须保留的语言 hierarchy 强基线。
- `2412.04453` **NaVILA** — 语言 waypoint + locomotion policy，实机成功率 88%；
  与非语言 `z_t/β_t` 做反事实对照。
- `2505.12705` **DreamGen** — world model 生成视频、IDM 恢复伪动作；
  rare-heavy 数据扩增候选，需防 hallucination × inverse-error 复合。
- `2501.15830` **SpatialVLA** — Ego3D encoding + adaptive action grids；
  可测 reference frame，但依赖外部几何估计。
- `2411.16537` **RoboSpatial** — 100 万图像、300 万空间 QA；
  是空间表征评估资产，不是闭环控制证据。

## 6. Axis E：脑科学、预测误差、睡眠与群体认知（14）

### A 级

- `10.1038/s41586-024-07851-w` **Cooperative Thalamocortical PE Circuit** — PE 选择性放大未预测
  感觉特征，而非显式传递“预测值减输入值”；支持局部 owner，不支持全局 curiosity 标量。
- `10.1038/s41593-023-01324-5` **Sleep Synchrony Enhances Consolidation** — 只有相位同步的
  PFC–MTL 慢波 / spindle–ripple 耦合增强次日记忆；sleep job 的 cadence 不是装饰。
- `10.1126/science.ado5708` **Hippocampal Balance of Reactivation** — CA2→CCK+ BARR 抑制
  SWR 过同步；反证 replay 越多越好，sleep owner 必须有稳态抑制。
- `10.1038/s41586-024-07973-1` **Human Hippocampal Temporal Structure** — 人类单神经元形成
  序列图、未来转移概率和压缩 replay；支持经验结构产生时间抽象。
- `10.1038/s41586-024-07799-x` **Abstract Representations in Human Hippocampus** —
  可观察变量与潜在上下文形成解耦几何，几何质量预测组合泛化；不证明几何就是因果 controller。
- `10.1038/s41593-024-01675-7` **Recurrent Planning and Replay** — meta-RL RNN 学会仅在有价值时
  触发 internal rollout，并复现人类思考时长和大鼠 replay。
- `10.1101/2023.11.28.569070` **Humans Balance Detailed and Abstract World Models** —
  环境可预测性降低时，人类减少时间抽象；抽象尺度应随 regime 调节。
- `10.1126/sciadv.adi4927` **Human Dopamine Reward/Punishment PE** — 奖励与惩罚 PE 出现在
  不同时间窗 / valence channel；反证单一 signed scalar 足够。
- `10.1126/sciadv.adq9684` **Dopamine Across Informational Domains** — dopamine 同时编码
  reward PE 与无价值 sensory PE；强确证 `PE ≠ reward`。

### B 级

- `10.1101/2023.06.23.546250` **Cognitive Model Discovery via Disentangled RNNs** —
  从行为恢复稀疏可命名 dynamics；解释模型是候选假说，不是真实模块证明。
- `10.1038/s41467-023-40141-z` **Experimental FEP Validation In Vitro** —
  特定培养神经网络沿 free-energy 下降；不可外推 sentience 或唯一产品 reward。
- `10.1038/s41562-025-02324-0` **Memory Shapes Human Reward Learning** —
  人类行为需要多个独立、丰富 memory variable；反证单一 value/RPE 总结历史。
- `10.1038/s41586-024-07126-4` **Bumblebee Social Learning** —
  群体传播可保留单体无法独立创新的程序结构；不证明 colony central controller。
- `10.3389/fnbeh.2025.1533372` **Ant Cooperative Clearing** —
  单蚁只凭局部信息素产生群体巢向清障；强支持局部快照耦合，不支持共享全局内部状态。

## 7. 跨轴裁决

### 7.1 现在最强的外部证据组合

- **R2**：DINO-WM + HPT/CrossFormer + SkillOS（前批）共同支持 frozen/shared substrate 与 owner-local
  adaptation；GR00T / Active Inference TTS 是端到端反例。
- **R3/R4**：ETA + Multi-Timescale Planning + LAPA/LAOM 成对构成“抽象可涌现，但必须检查可控性和
  distractor leakage”的完整证据链。
- **R5/R6**：SaliMory + DeltaMem + Beyond Perplexity + sleep synchrony/BARR 共同要求内容、频率、
  provenance、行为证据与稳态遗忘同时存在。
- **R-PE**：ICL Curiosity 负面定理 + thalamocortical PE + dopamine sensory PE 共同锁定：
  mismatch 是原始信号，reward / curiosity / evaluation 都是带假设的 readout。
- **R10/R15**：Falsifiable Gates + Deployment Simulation + AuditBench + CaMeL + MUSE/OpenUnlearning
  共同构成发布、运行时权限、审计、删除和回滚五层门。
- **R7/R11/R14**：PAHF + DynToM + Lifelong-SOTOPIA + longitudinal RCT + trust repair 共同说明：
  user/relationship state 必须动态、可修正、带健康边界，不能等同 persona 或互动时长。

### 7.2 需要新增的 kill conditions

1. latent action 在 distractor 下不再预测真实 action → kill “emergent action abstraction” claim。
2. hierarchy 生成 OOD subgoal 且 support constraint 无法恢复 → kill hierarchical planner promotion。
3. TTT 只降 perplexity、later free recall 仍为零 → kill deployment-memory claim。
4. audit tool 在 strategic deception 下系统失效 → kill “monitor 覆盖充分” claim。
5. trust / engagement 提升同时伴随依赖或真人社交下降 → kill relationship optimization。
6. unlearning 只过单一遗忘指标、隐私或 retained utility 失败 → kill deletion certificate。
7. sleep consolidation 只 replay 无抑制 / forgetting arm → kill slow-memory promotion。

## 8. 下载与阅读状态

- 28 篇核心增量：`research/papers/sweep-2607/`
- 78 篇广度扩展：75 篇开放 PDF 位于 `research/papers/frontier-map/`，3 篇下载受限论文保留 link-only
- 下载脚本：`research/download_frontier_map_expansion.sh`
- 下载汇总：`research/frontier-map-2024-2026-download-summary.md`
- 原始深读：`research/frontier-sweep-2026-07-20.md`

广度层采用“官方元数据核验 + PDF 关键机制 / 实验 / 限定复核 + 分层地图”口径；A 级论文进入后续
专题深读，B 级作为 baseline / negative anchor，C 级只保留链接和观察项，不因数量进入 runtime。

## 9. 一句话地图

当前前沿已不是“Transformer、memory、RL、world model 选哪个”的横向菜单，而是一组纵向接缝：
稳定基底之上学习什么 latent control、PE 如何变成有界 credit、记忆怎样跨频率巩固又可删除、关系怎样
在长期互动中增进而不制造依赖、以及所有自修改如何在可欺骗的监控环境中仍保持可证伪和可回滚。
