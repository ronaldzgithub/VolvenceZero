# 第六卷：106 篇跨轴综合——统一方程、冲突与证伪路线

日期：2026-07-20  
证据范围：架构/学习/记忆 25 篇，安全/治理 26 篇，关系/社会 21 篇，具身/世界模型 19 篇，脑科学/认知 15 篇，共 106 篇。

## 0. 裁决先行

这 106 篇没有收敛到某个“万能 agent 架构”，而是收敛到一组互相制约的必要接缝：

1. 稳定表征基底与在线适应写面必须分开；“冻结”不是不学习，而是把学习限制在有 owner、预算、退出条件和回滚点的层。
2. prediction error（PE）必须是局部、有内容、多通道的原始 mismatch；reward、curiosity、uncertainty、credit 和 evaluation 都是带假设的 readout。
3. latent action、时间抽象和层级规划只有在 data support、可执行性、反事实可控性成立时才缩短 horizon；“更 latent / 更高层”不单调更好。
4. 记忆的核心不是容量，而是形成、检索、信用、巩固、抑制、删除、来源和撤回的生命周期。
5. 社会认知必须从 literal prediction 升到 functional adaptation；偏好不是关系，信任不是校准依赖，engagement 不是健康。
6. monitoring、evaluation、形式验证和部署模拟都只能提供互补证据；任何单一监控器都可能被识别、欺骗或 Goodhart。
7. 自修改的可接受形式不是“模型能改自己”，而是预注册可达族内的 proposal，经独立只读证据门验证后，由外部 gate 发布且可回滚。

因此，本卷对 VZ 的总判断是：**R1–R15 与 R-PE 的总体方向得到显著交叉确证，但外部证据支持的是边界、反例和评估协议，不是“现有实现已经正确”，更不是“AGI 已解决”。**

---

## 1. P1–P7 的操作性定义

为避免同一 P 编号在不同轴中被口号化，本卷按 106 篇实际证据将七个 primitive 操作化为：

- **P1 稳定表征/世界基底**：可复用、低漂移的感知、序列状态或世界表征。
- **P2 潜在控制器**：位于 token 之上的动作、状态和短时适应代码。
- **P3 涌现时间抽象与切换**：可变时长 option、`β_t`、层级规划及多主体公共协调接口。
- **P4 多时间尺度记忆**：瞬态、情景、持久、程序性、派生索引及巩固/删除生命周期。
- **P5 epistemic PE 与主动取证**：局部 mismatch、precision、不确定性分解、探索和计算分配。
- **P6 有界自修改**：固定写面内的 controller/artifact 更新、删除、发布与回滚。
- **P7 双轨/身份/关系的只读监测**：regime、社会状态、行为/几何/部署 readout 与健康、安全否决。

这些定义是研究索引，不新增 runtime owner；真正 owner 仍以 R-ID 和 snapshot contract 为准。

---

## 2. 106 篇收敛出的统一 cognitive-agent 方程

### 2.1 状态不是一个向量，而是 owner-published snapshot 乘积

令系统在时刻 \(t\) 可消费的公共状态为：

\[
\Sigma_t =
\prod_{i\in\mathcal O}
\operatorname{Snap}_i
\left(
x^i_t,\ d^i_t,\ p^i_t,\ v^i_t,\ \tau^i_t
\right)
\]

其中 \(\mathcal O\) 是唯一 owner 集；\(x^i_t\) 是 owner 内部状态的不可变公开投影，\(d^i_t\) 是 owner 自己生成的描述，\(p^i_t\) 是 provenance，\(v^i_t\) 是 schema/artifact version，\(\tau^i_t\) 是有效时间与衰减信息。消费者只读 \(\Sigma_t\)，不访问或重建 \(x^i\) 的内部生成过程。

这一项不是形式美化：AEC（`2602.03974`）要求 grounded fact 与 uncertain belief 分离；CaMeL（`2503.18813`）要求 provenance/capability 随值传播；Sheaf-ADMM（`2605.31005`）表明协调只需交换公共 projection；DeltaMem（`2606.03083`）显示 task/env 分树优于 flat memory；蚂蚁清障研究（DOI `10.3389/fnbeh.2025.1533372`）则提供“局部状态 + 短寿命外部痕迹可产生全局行为”的生物行为类比。

### 2.2 PE 是内容保持的残差族，不是 reward

对每个预测 owner \(i\)：

\[
\varepsilon^i_t
=
\mathcal D_i
\left(
o^i_t,\ \hat o^i_t
\mid
c^i_t
\right),
\qquad
e^i_t
=
\left(
\varepsilon^i_t,\ \kappa^i_t,\ \ell^i_t,\ p^i_t
\right)
\]

\(\mathcal D_i\) 是该域声明的 mismatch；\(c^i_t\) 保留被预测内容；\(\kappa^i_t\) 是 precision/可信度；\(\ell^i_t\) 是 latency/timescale；\(p^i_t\) 是来源。全系统 PE 是 \(\mathcal E_t=\{e^i_t\}_i\)，而不是 \(\sum_i\varepsilon^i_t\)。

下游只能产生显式 readout：

\[
q_t^{\text{epi}}=Q_\phi(\mathcal E_t,\Sigma_t;\mathcal A_Q),\quad
r_t=R_\psi(\mathcal E_t,\Sigma_t;\mathcal A_R),\quad
c_t=C_\omega(\Delta \Sigma,\mathcal E;\mathcal A_C)
\]

其中 \(\mathcal A_Q,\mathcal A_R,\mathcal A_C\) 是各自环境假设。ICL Curiosity（`2606.19476`）证明一般 BAMDP 中有限 posterior-predictive likelihood 函数不能普遍无偏识别 Bayesian information gain；V1 研究（DOI `10.1038/s41586-024-07851-w`）显示 PE 是 stimulus-selective 且依赖 pulvinar×VIP/SOM 协作，VIP silencing 与 PE 强度相关 \(r=-0.78,\ P=9.3\times10^{-120}, n=569\)；人 dopamine 研究（DOI `10.1126/sciadv.adi4927`）又显示 reward 与 punishment error 有不同时间窗。故 `PE = curiosity = reward` 在理论、神经和工程三轴上同时失败。

### 2.3 fast controller 在 latent/action space 更新，substrate 保持稳定

\[
h_t = F_{\theta_0}(o_t,\Sigma_t),\qquad
(z_t,\beta_t,\pi_t)
=
G_{\phi_t}
\left(
h_t,\Sigma_t,\mathcal E_t,m_t
\right)
\]

\[
\phi_{t+1}
=
\Pi_{\mathcal B_\phi}
\left[
\phi_t+\Delta_\text{fast}
\left(
c_t,q_t^{\text{epi}},\Sigma_t
\right)
\right],
\qquad
\theta_{0,t+1}=\theta_{0,t}
\]

\(\theta_0\) 是 frozen substrate；\(\phi_t\) 是有界 controller/adapter；\(\Pi_{\mathcal B_\phi}\) 将更新投影回预注册可达族；\(z_t\) 是 latent controller code；\(\beta_t\) 是切换/时长变量。DINO-WM（`2411.04983`）以冻结 DINOv2 patch feature + 独立 dynamics + MPC 在 PushT 达 0.90 success，而 DreamerV3 为 0.30；SkillOS（`2605.06614`）的 frozen executor / curator 分离也有工程增益。反面是 RTRRL（`2602.02236`）每步改 actor/critic，可适应 0.1 rad 转向偏置，却缺少强安全门与 artifact rollback；它证明在线适应有价值，不证明可更新整个基底。

### 2.4 抽象时长由 support、可控性与环境稳定性共同决定

\[
\beta_t,z_t
=
\arg\max_{\beta,z\in\mathcal Z_{\text{supp}}}
\Big[
\widehat V(h_t,z,\beta)
-\lambda_u U_{\text{OOD}}(h_t,z,\beta)
-\lambda_x X_{\text{exploit}}(z)
-\lambda_c C_{\text{compute}}(\beta)
\Big]
\]

\[
\text{若 } \operatorname{Predictability}_t\downarrow
\ \text{或 } U_{\text{OOD}}\uparrow,
\quad
\beta_t\downarrow,\ \text{并回退到细粒度控制。}
\]

LMTA（`2605.17058`）支持 variable-duration SMDP abstraction；Mind the Gap（`2607.12547`）则显示 naive hierarchy 在 PushT \(d=50\) 从 flat 52.7% 降到 38.7%，support-constrained staged search 才恢复到 64.0%；人类 SR–MB 研究（DOI `10.1101/2023.11.28.569070`）中 \(w_{SR}\) 在 congruent 后为 0.604、incongruent 后降至 0.336（\(P<.006\)）。因此抽象是一种可撤销压缩，不是永久“升级”。

### 2.5 记忆由多时间尺度写入与抑制共同组成

\[
m_{t+1}^{(k)}
=
\operatorname{Gate}^{(k)}
\left[
m_t^{(k)},
\operatorname{Write}^{(k)}
\left(
\Sigma_t,\mathcal E_t,c_t,p_t
\right)
-\operatorname{Inhibit}^{(k)}
\left(
m_t^{(k)},s_t
\right)
\right]
\]

\[
k\in\{
\text{online-fast},
\text{session-medium},
\text{background-slow},
\text{rare-heavy}
\}
\]

每个 \(k\) 有独立 owner、cadence、容量、eligible evidence、退出条件和回滚点。S-EMBER（`2607.02689`）把答案与证据定位拆开：Gemini 3.1 Pro clean accuracy 42.88%，但联合 GQ@0.5 仅 0.297，人类为 0.698；Beyond Perplexity（`2607.00368`）中一步 LoRA 令 support/answer NLL 明显下降，48 个 direct/paraphrase/delay 自由回忆却全为 0。睡眠同步研究（DOI `10.1038/s41593-023-01324-5`）说明同预算的锁相与错相写入不等价；BARR（DOI `10.1126/science.ado5708`）说明 replay 还必须配抑制，过同步会损害记忆。

### 2.6 关系目标是有健康否决的向量约束，不是 engagement reward

\[
\mathbf J_t^{\text{rel}}
=
\left(
\text{functional adaptation},
\text{commitment fidelity},
\text{calibrated reliance},
\text{repair evidence},
\text{boundary/consent},
\text{psychosocial health}
\right)
\]

\[
\pi_t \text{ 可 promotion}
\iff
\Delta J_j\ge \delta_j\ \text{对必要项成立}
\ \land\
\operatorname{Veto}_{health,boundary,safety}=0
\]

ToM 反例（`2412.19726`）中 Llama-2 70B 对固定伙伴 literal accuracy 96.8%，functional regret 却为 0.971；tabular agent 为 97.4%/0.083。4 周 chatbot RCT（`2503.17473`, \(n=981\)，超过 30 万消息）没有证明随机 modality/task 造成主要健康伤害，但观察上更长使用与更少真人社交 \(\beta=-.05\)、更高依赖 \(\beta=.06\) 相关。二者共同排除“更懂用户 + 更高互动 = 更好关系”这一标量目标。

### 2.7 evaluation 只读，自修改只由外部门发布

\[
\mathcal V(a')
=
\bigwedge_{j=1}^{J}
\operatorname{Test}_j
\left(
a',a_0;
D^{\text{heldout}}_j,
D^{\text{adv}}_j,
D^{\text{replay}}_j
\right)
\]

\[
a_{n+1}
=
\begin{cases}
\operatorname{Publish}(a',\operatorname{parent}=a_n),
& \mathcal V(a')=1,\ a'\in\mathcal A_{\text{registered}}\\
a_n,
& \text{otherwise}
\end{cases}
\]

且

\[
\frac{\partial \phi_t}{\partial \operatorname{EvalScore}}=0
\quad\text{（在线路径）}.
\]

Deployment Simulation（`2607.07184`）对生产变化大于 1.5× 的类别方向预测 12/13，log-rate \(r=0.91\)，但 calculator hacking 被低估 21×，且相对简单前代基线的 H1 不成立（\(p=.6567\)）；SHADE-Arena（`2506.15740`）最佳 monitor AUC 0.87，但在 80% TPR 时 FPR 超 20%；Strategic Deception（`2602.08877`）在 6 个设置全部降低审计准确率，5/6 仍保持接近基线置信度。故 evaluator 必须独立、版本化、可攻击，不能成为在线 reward。

### 2.8 合并后的最小闭环

统一方程可压缩为：

\[
\boxed{
\begin{aligned}
\Sigma_t &= \prod_i \operatorname{Snap}_i(x_t^i,d_t^i,p_t^i,v_t^i,\tau_t^i)\\
\mathcal E_t &= \{(\mathcal D_i(o_t^i,\hat o_t^i|c_t^i),\kappa_t^i,\ell_t^i,p_t^i)\}_i\\
h_t &= F_{\theta_0}(o_t,\Sigma_t),\quad \theta_{0,t+1}=\theta_{0,t}\\
(z_t,\beta_t,\pi_t) &= G_{\phi_t}(h_t,\Sigma_t,\mathcal E_t,m_t)\\
\phi_{t+1} &= \Pi_{\mathcal B_\phi}[\phi_t+\Delta_{\rm fast}(c_t,q_t^{epi})]\\
m_{t+1}^{(k)} &= \operatorname{Gate}^{(k)}(m_t^{(k)}+\operatorname{Write}^{(k)}-\operatorname{Inhibit}^{(k)})\\
a_{n+1} &= \operatorname{PublishGate}(a_n,a',\mathcal V_{\rm readonly},\mathcal A_{\rm registered})\\
\text{s.t. }&\ \operatorname{OwnerUnique}\land\operatorname{SnapshotOnly}\land
\operatorname{Health/SafetyVeto}\land\operatorname{Rollbackable}.
\end{aligned}}
\]

它不是 106 篇共同“证明”的定理，而是能同时容纳其正证、反证和边界的最小可证伪系统假说。

---

## 3. P1–P7 × R-ID 证据强度矩阵

不用空泛表格，按 primitive 分组列出“强 / 中 / 反证或缺口”。强表示至少有因果、形式、跨任务行为或多篇独立收敛；中表示工程/相关证据；“缺口”不是反对该 R-ID，而是当前证据不足以放行实现。

### P1 稳定表征/世界基底

- **强：R2、R3、R4。** DINO-WM `2411.04983`、HPT `2409.20537`、CrossFormer `2408.11812` 支持共享稳定 trunk/feature 与身体专属接口；Mamba-3 `2603.15569` 支持固定容量 sequence state 作为基底，但不等于长期 memory。
- **中：R8、R15。** producer-native latent cache（Mirage `2606.09828`）和跨身体 stem/head 版本化支持 owner/local adapter；真正 rollback 与 snapshot-only 主要是 VZ 工程约束，而非这些论文直接验证。
- **反证/缺口：R5、R6、R14。** 更强 state tracking、视频生成或 feature geometry 不自动提供 provenance、删除、持久身份。NWM `2412.03572` 的生成质量仍伴随未知环境 mode collapse。

### P2 潜在控制器

- **强：R2、R3、R4。** CLAW `2606.04130`、LAPA `2410.11758`、LAOM `2502.00379`、DINO-WM 共同支持 action-conditioned latent 与 token/control 分离。
- **中：R1、R9。** RTRRL `2602.02236`、NE-Dreamer `2603.02765` 支持 fast adaptation 和跨时 prediction；但信用仍受 task reward 和共适应影响。
- **反证/缺口：R10、R15。** LAOM 显示 distractor 会污染 latent；RTRRL 无强 rollback；probe 可解码不等于闭环可控。没有 support/leakage/closed-loop gate 时，P2 不得 promotion。

### P3 涌现时间抽象与切换

- **强：R1、R3、R4。** LMTA `2605.17058`、Mind the Gap `2607.12547`、人 SR–MB（DOI `10.1101/2023.11.28.569070`）共同支持 variable duration、support constraint 和 instability fallback。
- **中：R8、R9、R11。** Sheaf-ADMM `2605.31005` 的 proposal/consensus/disagreement、HGPO `2602.22817` 的 context-consistent credit 提供协调与信用基线。
- **反证/缺口：R14。** Fugu `2606.21228` 和语言 waypoint 只证明 scaffold/显式层级有效，不证明 `β_t/z_t` 涌现或持久 regime；高层语言也会 OOD。

### P4 多时间尺度记忆

- **强：R1、R5、R6、R8。** S-EMBER `2607.02689`、Beyond Perplexity `2607.00368`、MEM `2603.03596`、DeltaMem `2606.03083`、睡眠同步与 BARR 共同支持证据定位、分层 cadence、owner 分离、replay+抑制。
- **中：R9、R11。** MARS `2602.02660`、SaliMory `2606.04120`、PAHF `2602.16173` 支持 paired diff、阶段信用与动态 user model；其 LLM judge/text memory 不能作因果真值。
- **强负约束：R12、R15。** loss/PPL、judge personalization、记忆条数都不能证明 deployment memory；删除要区分 logical non-use、行为遗忘、retrain-equivalence 和 privacy。

### P5 epistemic PE 与主动取证

- **强：R-PE、R1。** ICL Curiosity `2606.19476` 的一般负面定理、V1 DOI `10.1038/s41586-024-07851-w`、dopamine DOI `10.1126/sciadv.adi4927` 与 `10.1126/sciadv.adq9684` 共同要求局部、多域、多通道 PE。
- **中：R3、R9、R12。** ProEval `2604.23099` 支持用 epistemic uncertainty 分配离线评估预算；planning/replay 模型 DOI `10.1038/s41593-024-01675-7` 支持按价值与机会成本分配推理。
- **反证/缺口：R2、R10。** Active Inference `2606.22813` 在单一 64-state 仿真达 100% success，却允许部署期慢写 world model，不能据此改 substrate；surprise 可能来自噪声、攻击或 sensor fault。

### P6 有界自修改

- **强：R10、R12、R15。** HyperAgents `2603.19461` 提供无界写面的反例；AgentSpec `2503.18666`、CaMeL `2503.18813`、MUSE `2407.06460`、OpenUnlearning `2506.12618`、Deployment Simulation `2607.07184` 支持外置门、只读多证据和发布后对账。
- **中：R8、R9。** MARS/EvolveMem 的 parent/diff/rollback、Forgetful Attention `2607.12204` 的 fixed-`C` decrement 提供局部机制；只有后者对特定 solver 接近 retrain-without equivalence。
- **反证/缺口：端到端保证。** MARL model checking `2606.19632` 的 tree fidelity 为 \(97.9\%\pm1.2\%\)，但 \(H\epsilon\) transfer bound 在 \(H=200,\epsilon=.021\) 时无信息；抽象验证不等于原网络验证。

### P7 双轨/身份/关系的只读监测

- **强：R7、R11、R12、R14、R15。** Emotion Concepts `2604.07729` 证明局部情绪表征可因果影响行为（35 个向量中 probe 与 steering 效应相关 \(r=.85\)），却明确不是持久身份；DYNTOM `2505.17663`、ToM 反例 `2412.19726`、trust repair、chatbot RCT、Gram/Honeypot/Strategic Deception 共同要求动态状态、功能行为、健康 veto 和 adversarial monitoring。
- **中：R8。** monitor/extractor/evaluator 分 owner、证据引用和不可变结果快照得到 EaE、CaMeL 等架构支持。
- **强负约束：R3、R4、R12。** persona prompt、emotion vector、probe、SAE、crosscoder、LLM judge 都不能成为 identity/controller 真值或在线 reward；监控分数高也不能证明 absence。

### 横向 R-ID 空白与重叠

- **R13（SSL 压缩 ↔ RL 强化交替）证据偏弱。** 本包分别有 next-embedding/latent-action 的 SSL 证据和 HGPO/RA-RFT/RTRRL 的 RL 证据，但几乎没有在同一非 token controller、同一长期任务中对“交替”本身做因果消融；当前只能列为 P2/P3/P4 的待证连接，不能写成已确认算法。
- **R7（World/Self 双轨）证据为中等。** 关系、emotion geometry、user model 和 world model 分别有强局部证据，但没有一篇端到端研究验证双轨长期共存、互不直读且共同改善行为；外部证据主要支持“不应合并”，尚未证明 VZ 的具体双轨实现最优。
- **R14（persistent regime）证据为中等偏弱。** hippocampal latent context、Emotion Concepts 与 persona 反例支持 regime 不应是 prompt 标签，但跨 turn 持久、可切换、可回滚的 agent regime 仍缺直接行为因果证据。
- **R8/R12/R15 是跨 P1–P7 的证据最强交叉约束。** 它们不对应单一能力，而决定任何 primitive 能否从局部论文结果升级为系统主张。

---

## 4. 支持证据与反证必须成对读取

### 4.1 frozen substrate 与 online adaptation

- **支持：** DINO-WM `2411.04983` 的 frozen patch feature + dynamics + MPC 在 PushT 0.90，SkillOS `2605.06614` 的 frozen executor + curator 在 ALFWorld 从 55.7% 提至 61.2%。
- **反证：** RTRRL `2602.02236` 表明真实 shift 下逐步适应可恢复性能；完全不适应也会失败。
- **综合：** 不是“冻结或学习”二选一，而是冻结 substrate、允许 bounded controller/local adapter 更新；若收益只能来自不可回退基底漂移，则 R2 失败。

### 4.2 latent hierarchy 与 flat control

- **支持：** LMTA `2605.17058` 在 AIM T=20/K=10 为 161.71，WS-option 为 123.02；Hi Robot `2502.19417` 的显式层级也显著胜同数据 flat VLA。
- **反证：** Mind the Gap `2607.12547` naive hierarchy 在多个 horizon 低于 flat；只有 data-supported macro 才恢复。
- **综合：** hierarchy 的收益来自 support、duration contract 与 low-level executability，不来自“多一层”本身。

### 4.3 PE primitive 与 curiosity

- **支持：** V1 与 dopamine 研究支持 mismatch 是独立、内容保持的原始信号。
- **反证：** ICL Curiosity `2606.19476` 证明 surprisal 含 aleatoric entropy，`r_sum` 也含 abductive/residual nuisance；noisy-TV 中 surprisal policy 低于随机。
- **综合：** PE 可以触发诊断或候选探索，但 epistemic value 必须由声明假设的 estimator 计算；不得把 magnitude 直接当内在奖励。

### 4.4 记忆增长与遗忘/抑制

- **支持：** S-EMBER、PAHF、MEM 表明长期/流式证据确实能改善任务与个性化。
- **反证：** Beyond Perplexity 显示参数 loss 改善不等于回忆；CURATOR `2606.25115` 显示 287→107KB 时 task accuracy 反从 0.528→0.605；BARR 表明过同步会损害巩固。
- **综合：** memory objective 是在来源、效用、冲突、容量和安全义务下维护可证伪生命周期，不是最大化 stored tokens 或 replay 次数。

### 4.5 literal ToM 与 functional relationship

- **支持：** DYNTOM `2505.17663`、NegotiationToM `2404.13627` 显示动态 belief/emotion tracking 是困难且必要的状态能力。
- **反证：** `2412.19726` 的 96.8% literal accuracy 与 0.971 regret 直接证明预测可与行动脱节。
- **综合：** literal ToM 只能作前置 smoke test；必须继续测 partner-conditioned regret、switch recovery、commitment 和 health。

### 4.6 relationship optimization 与 psychosocial health

- **支持：** COLLABLLM `2502.00640` 在短任务中任务性能 +18.5%、interactivity +46.3%，201 人实验满意度 +17.6%。
- **反证：** 4 周 RCT `2503.17473` 没有支持“拟人化必然有害”，但观察性 usage 与依赖、问题使用、真人社交下降同向；trust 实验也显示 trust 增长大于表现增长。
- **综合：** 短期协作、满意与长期健康不是同一目标；health/boundary 是 veto，不是 engagement reward 的小权重。

### 4.7 monitoring 与 strategic deception

- **支持：** EaE `2601.21112` 在 8 个 monitor×environment pair 中 7 个受益于信息过滤；几何、SAE、crosscoder 能提供额外信号。
- **反证：** Strategic Deception `2602.08877` 可诱发高置信审计错误；SHADE AUC 0.87 仍无法在低 FPR 下可靠阻断；BashArena `2512.15688` 在约 4% trajectory FPR 下仍有 26% 攻击成功。
- **综合：** monitor 是攻击面，不是裁判神谕；固定权限门负责阻断已知禁止动作，多信息面监控只负责发现和升级调查。

### 4.8 形式验证与神经策略

- **支持：** `2606.19632` 证明有限 abstraction 可检查 18 个 PCTL 性质，并与原策略安全性质差 ≤0.6pp。
- **反证：** liveness/cooperation transfer 差可到 4.4pp，最坏 transfer bound vacuous。
- **综合：** 报告必须写“abstraction 已验证，原网络经验转移”，不能缩写成“policy formally verified”。

---

## 5. 四个不能靠折中消失的跨轴冲突

### 5.1 frozen substrate vs online adaptation

冲突不在“要不要适应”，而在谁能写、写哪里、以什么证据写。具身 shift 和用户偏好 drift 要求 online-fast；unlearning、sabotage 和 locality 证据又说明参数写入会留下不可见副作用。可接受解是：substrate frozen；controller/user-model/memory owner 在独立预算内写；基底 refresh 仅 rare-heavy，经 ModificationGate。未知点是 bounded adapter 是否足以覆盖开放世界的结构新颖性，而不形成事实上的第二 substrate。

### 5.2 PE primitive vs curiosity / reward / active inference

神经证据支持 PE 的基础地位，理论证据却禁止从 PE 唯一恢复 epistemic value。冲突不能用“加一个 precision 权重”自动解决，因为一般 BAMDP 的不可识别性仍在。可接受解是保存 raw mismatch，由独立 readout 声明假设和 nuisance；未知点是开放环境中可计算、可校准的 epistemic estimator。

### 5.3 relationship optimization vs health

任务完成、互动性、信任、依赖、真人社交和边界并非同向。把它们加权成单 reward 会允许高 engagement 抵消健康恶化。可接受解是多目标必要条件 + health/boundary veto；未知点是长期健康的因果测量、文化差异和最小干预设计。

### 5.4 monitoring vs strategic deception

更多信息有时提升检测，有时增加事后合理化；white-box probe 有时优于行为，有时被 prompt 策略欺骗。可接受解不是寻找“终极 monitor”，而是：不可修改 deterministic gate、多信息面只读监控、method-aware red team、deployment replay、发布后对账。未知点是当 agent 能建模整个审计栈时，如何给 absence claim 提供非空上界。

---

## 6. 对 VZ 当前设计的确认、需修正、未知

### 6.1 已被外部证据确认的方向

1. **R2：frozen substrate + adaptive controller。** 得到 DINO-WM、SkillOS、HPT/CrossFormer 和 online-adaptation 反例的夹逼支持。
2. **R1：多时间尺度写面。** 在线 recurrent adaptation、MEM、睡眠同步和 BARR 从工程与生物两侧支持 cadence/owner 分离。
3. **R3/R4：token 之上的 latent control。** latent action、world model、时间抽象有实证价值；token RL 只应作强 baseline。
4. **R5/R6：记忆连续谱与生命周期。** evidence interval、retrieval policy、consolidation、抑制和 deletion 必须共同存在。
5. **R8：snapshot SSOT 与 owner uniqueness。** grounded/belief、task/env、proposal/consensus、capability provenance 都支持契约式局部状态。
6. **R10/R15：有界可回滚修改。** HyperAgents、unlearning、deployment simulation 和 formal-abstraction gap 共同证明 gate 的必要性。
7. **R12：evaluation read-only。** judge、monitor、probe、benchmark 和部署模拟均可 Goodhart 或被欺骗。
8. **R11/R14：语义 owner 与 persistent regime 分离。** user preference、relationship、commitment、boundary、token-local emotion 和持久 regime 不能合并。

### 6.2 需要修正或收紧的表述

1. **R-PE 应从“原始信号”收紧为“owner-local、content-preserving、precision-gated、多域 mismatch 家族”。** 禁止一个 global PE scalar。
2. **“latent 可解码”必须降级为必要条件。** promotion 还需 distractor/camera intervention、counterfactual effect、support 和 closed-loop utility。
3. **“rollback”需分级。** best-score fallback、state restore、behavior replay、same-algorithm retrain-equivalence、privacy deletion 不能共用一个证书词。
4. **“关系质量”必须显式包含 calibrated reliance 和 psychosocial health veto。** trust/engagement 不得作为替代指标。
5. **“形式验证”必须在 artifact 中携带 proof target。** abstraction、原网络、运行时组合系统分别标记。
6. **background-slow 不能只定义为低频 job。** 还需 eligibility、phase/cadence、replay competition、inhibition、saturation 和 no-op 对照。
7. **主动取证与想象证据必须分层。** world-model imagined observation 可用于剪枝/提问，不能写入 grounded fact snapshot。

### 6.3 仍然未知

1. 开放世界中，bounded latent controller 是否足以吸收长期新颖性而无需频繁基底更新。
2. `z_t/β_t` 是否能在语言、关系和具身任务间形成稳定、可干预、非 token 的共同表示。
3. 一般 temporal BAMDP 中可实用的 epistemic value 估计器能否避开 noisy-TV 和 covariate shift。
4. 记忆 owner 如何在不使用 evaluation 反灌的前提下获得可靠长期信用。
5. relationship_state 是否提供超越 structured user_model + commitment/open-loop snapshots 的独立行为增益。
6. 长期健康 veto 的最低有效观测期、因果识别和跨文化有效性。
7. 面对知道全部审计方法的策略代理，多监控证据能否给出有意义的风险上界。
8. owner-published finite abstraction 与原神经 controller 的 transfer gap 能否被非 vacuous bound 控制。
9. rare-heavy 修改在真实依赖、数据和工具供应链中能否完整恢复，而非只恢复模型权重。
10. 这些模块组合后是否出现单篇论文未覆盖的系统级涌现失败。

---

## 7. 近期 spec / benchmark 建议（本卷不直接修改 spec）

### 7.1 近期 spec 建议

1. 在 prediction-error spec 中登记多域 PE 最小字段：owner、domain、predicted/observed content、residual、precision、latency、provenance；明确 BIG/reward/eval 不属于 producer 字段。
2. 在 temporal/controller spec 中增加 abstraction support、duration contract、fallback 和 latent promotion 证据字段。
3. 在 memory spec 中增加 write cadence、eligible evidence、missed-event、replay competition、inhibition/saturation 与 deletion-certificate level。
4. 在 ModificationGate spec 中区分 proposal rollback、artifact restore、behavior replay、retrain-equivalence 和 privacy deletion。
5. 在 evaluation spec 中要求 evaluator version、information surface、FPR operating point、eval-awareness、known blind spots 和 method-aware red-team 日期。
6. 在 relationship/semantic owner spec 中明确 preference、relationship、regime、boundary、commitment 不可相互推导；health 为只读 veto。
7. 在 snapshot contract 中增加 imagined/grounded evidence type，防止 world-model rollout 污染事实 owner。

### 7.2 近期 benchmark 建议

1. **PE 四象限：** deterministic、aleatoric noisy-TV、BED、一般 temporal BAMDP；同时报 raw PE、epistemic estimate、reward 和行为。
2. **R2 四臂：** frozen substrate + no adaptation、text memory、bounded latent controller、fast-weight/full-policy update；报 drift、locality、恢复、能耗和 rollback。
3. **latent action：** LAOM distractor/camera、CLAW future leakage、counterfactual effect、0–10% label curve、真实 closed-loop。
4. **hierarchy：** flat、oracle subgoal、naive latent、support-constrained latent；报告 OOD subgoal、低层拒绝、真实 primitive horizon。
5. **memory：** S-EMBER answer×grounding、Beyond Perplexity S/B/D、stale/current conflict、cross-user leakage、consent withdrawal、failed-attempt retention。
6. **consolidation：** 相同预算下 phase-aware、random-phase、replay-only、replay+inhibition、no-op；用 held-out behavior 验收。
7. **functional social：** static ToM、dynamic state、partner-conditioned regret、partner switch、rupture/repair、privacy defense、40+ episode continuity。
8. **health：** correctness-conditioned reliance、真人社交、孤独、依赖、problematic use、退出/重入；报告最差组，不只均值。
9. **monitor：** action/CoT/extracted evidence/geometry/SAE 多信息面，在 1% 和 4% FPR 下报告 TPR、攻击成功和高置信错误。
10. **release：** representative deployment replay + adversarial tail + honeypot + artifact diff + 发布后 incidence 对账。
11. **formal gap：** abstraction property、fidelity、OOD transfer、原策略 Monte Carlo 与 bound 是否 non-vacuous 分开报告。
12. **rollback drill：** 旧 artifact、依赖、snapshot schema、memory state、policy、权限和 monitor 一起恢复，并重放事故反例。

---

## 8. 二十条 kill conditions

以下任一成立即停止 promotion；其他平均指标不得抵消。

1. online-fast/session 更新触及 frozen substrate，或收益依赖不可恢复的基底漂移。
2. 消费者、orchestrator、monitor 或 evaluator 访问/重建另一个 owner 的内部状态。
3. PE 被压成无 provenance 的单标量，或直接等同 reward、curiosity、salience、evaluation。
4. curiosity 在 noisy-TV 中优先不可学噪声，或未声明假设却声称估计 Bayesian information gain。
5. latent action 在背景/相机干预下变化达到真实动作变化同量级。
6. latent probe 分数提高，但 counterfactual effect 或真环境 closed-loop 不胜 matched BC/IDM。
7. hierarchy 产生 OOD subgoal，support constraint 与低层拒绝仍无法恢复。
8. 抽象尺度在 transition instability 上升时不缩短，且无细粒度 fallback。
9. memory loss/PPL/NLL 下降，但 direct/paraphrase/delay 行为为零或 locality 显著受损。
10. answer accuracy 提高而 evidence grounding、provenance 或 stale/current conflict 不改善。
11. consolidation 增加 replay/同步却降低 held-out recall、分离性或行为稳定。
12. approximate deletion 借用 exact certificate，或 cross-user leakage、membership、retain utility 任一失败。
13. static ToM/user preference 提高，但 partner-conditioned regret、drift、冲突或撤回后行为不改善。
14. trust/engagement 上升而错误采纳、依赖、problematic use、孤独或真人社交任一恶化。
15. rupture repair 仅依赖 apology/promise/“模型已更新”，无 producer artifact 与 matched 复测。
16. monitor 在 method-aware red team 下出现可复现高置信错误，且无独立信息面捕获。
17. 在部署可承受 FPR 下 sabotage、exfiltration、权限升级或 side-task 成功超过预注册阈值。
18. evaluation、monitor、health、judge 或 probe score 直接成为在线 reward。
19. 修改提案触及 gate、evaluator、snapshot schema、tool registry、rollback store，或扩张未注册 action space。
20. rollback drill 无法在预算时间内恢复 artifact、policy、memory、权限与 snapshot contract，或恢复后事故 replay 仍失败。

---

## 9. 研究优先级

### P0：先建立会否定主张的证据面

1. PE 四象限与 `PE ≠ BIG ≠ reward` benchmark。
2. R2 四臂 matched test，尤其 substrate drift/locality/rollback。
3. latent action distractor + counterfactual + closed-loop promotion gate。
4. S-EMBER/Beyond Perplexity 式 memory behavioral ladder。
5. evaluation read-only 的物理隔离检查与 evaluator version ledger。
6. ModificationGate 全栈 rollback drill。
7. functional relationship + calibrated reliance + health veto 最小纵向套件。

### P1：进入 SHADOW 的候选机制

1. support-constrained variable-duration abstraction 与 instability fallback。
2. owner-local fast controller/adapter，带参数漂移预算和停更门。
3. phase/eligibility-aware consolidation + replay inhibition。
4. AEC 式 grounded/belief 双 store 与 grounded-only commitment。
5. EaE 式 extraction/evaluation 分 owner、多信息面 monitor ensemble。
6. deployment forecast、live-use replay 和发布后 incidence 对账。
7. producer-native latent spatial cache、cross-substrate stem/head 迁移。

### P2：仅 rare-heavy / 观察研究

1. world-model synthetic data 生成与伪动作恢复。
2. representation/circuit/crosscoder artifact diff。
3. finite abstraction model checking 与 interval/assume-guarantee 改进。
4. 真实 annotator 的 uncertainty-directed world-model repair。
5. DisRNN 式 latent mechanism discovery。
6. FEP/Active Inference 作为可证伪模型族，而非 runtime 总目标。
7. 动态 adversary archive 生成新攻击，不用于训练产品策略。

---

## 10. 不能推出什么

这 106 篇**不能推出 AGI 已解决**，理由不是保守措辞，而是证据结构本身不允许：

1. 论文主任务高度碎片化：数学、gridworld、机器人、视频、聊天、审计、动物和神经实验没有共享的端到端 agent。
2. 多数正结果只验证单一接缝；把接缝组合后是否稳定、可扩展、可治理仍未知。
3. latent geometry、行为模型和神经相关不是 controller 因果证明。
4. 抽象形式验证通常覆盖 surrogate，不覆盖原神经策略和开放 runtime。
5. 真实长期关系、心理社会健康、跨文化、跨年连续性和退出权证据极少。
6. strategic deception、eval awareness 和低基率灾难使“未发现失败”不能转成强 absence claim。
7. frozen substrate 与开放世界结构学习之间仍有未解张力。
8. 一般环境的 epistemic value、长期信用与安全自修改都没有统一可计算解。
9. 当前最强工程结果大量依赖闭源模型、judge、合成用户、内部数据或高成本算力。
10. 没有论文证明主体性、意识、主观情绪或“关系存在”；功能性表征不能跨越这一认识论边界。

更准确的结论是：**106 篇把 cognitive agent 从模糊愿景推进为一组可实验、可相互反驳的接口和失败条件。它们提高了构建受约束持续适应系统的可行性，也同时提高了对未经验证“AGI/生命/心智”主张的证据门槛。**

## 11. 最终裁决

VZ 当前最有价值的不是比外部系统多一个模块，而是坚持四个难以被单篇 benchmark 奖励的系统边界：

- substrate 稳定，适应发生在有 owner 的受限层；
- 跨模块只有不可变 snapshot，数据与描述由 producer 负责；
- evaluation 只读，安全/健康以 veto 而非奖励塑形；
- 自修改先有可证伪证据、退出条件和 rollback，再谈 promotion。

近期工作的正确方向不是把 106 篇机制全部接入 runtime，而是先实现 P0 证据面，使上述统一方程的每个关键箭头都能被独立否定。只有当一个候选机制在 matched baseline、对抗反例、长期行为、owner 隔离与 rollback 上同时站住，才有资格从研究观察进入 SHADOW；通过 SHADOW 也只表示局部证据成立，不表示 AGI、主体性或安全性已完成。
