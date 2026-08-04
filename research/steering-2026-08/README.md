# Steering / 表征干预文献研究（2026-08）

## 缘起

Stage-3 权威扫（36/36，2026-08-04）正式判 `kill-eta`（operationalization-scoped，非理论普遍证伪），
主线转向"**读残差 + 有界 steering + Internal RL 学干预策略**"（[stage3.md](../../.cursor/plans/stage3.md) 方案 A）。
但转向的第一道生死门——**S2 因果 steering**——刚刚 FAIL：
S1 probe 在 layer20/896 维上把子目标读到 heldout `0.9833`（近天花板、且裸基底免费），
S2 沿该 probe 轴施加干预却拿到 target-plus vs noop = `-0.00072`（95% CI `[-0.01787, 0.01809]`），五门全败。

**"可读却不可扳"**是本轮最关键、也最反直觉的现象。本研究下载并深读 2024–2026 的 steering / 表征干预主线文献，
目的只有一个：**判定"可读却不可扳"是我们实现的偶然缺陷，还是学界已知的系统性现象；若已知，学界给出的修法是什么。**

结论（详见 [02_VZ_IMPLICATIONS.md](02_VZ_IMPLICATIONS.md)）：**这是学界已充分刻画的已知失败模式，且有可直接落地的预检与修法。**
我们 S2 复现的是学界的负结果，不是发现了新死路。

## 论文清单

下载脚本 [download_steering_2608.sh](../download_steering_2608.sh)，SHA 与状态见 [download-summary.md](download-summary.md)。已对 `research/**` 去重（9 篇均无重叠）。

| # | 论文 | 场次 | 本地 PDF | 服务的 screen |
|---|---|---|---|---|
| A1 | Understanding (Un)Reliability of Steering Vectors ([2505.22637](https://arxiv.org/abs/2505.22637)) | ICLR 2025 WS | `papers/understanding-unreliability-of-steering-vectors-2505.22637.pdf` | **S2 归因（核心）** |
| A2 | Steering off Course ([2025.acl-long.974](https://aclanthology.org/2025.acl-long.974/)) | ACL 2025 | `papers/steering-off-course-reliability-challenges-acl2025.pdf` | S2 跨模型泛化 |
| A3 | FaithSteer-BENCH ([2603.18329](https://arxiv.org/abs/2603.18329)) | 2026 | `papers/faithsteer-bench-deployment-stress-test-2603.18329.pdf` | 部署级警醒 |
| B1 | Contrastive Activation Addition / CAA ([2312.06681](https://arxiv.org/abs/2312.06681)) | ACL 2024 | `papers/contrastive-activation-addition-caa-2312.06681.pdf` | **S2' 方向来源** |
| B2 | Activation Steering via Generative Causal Mediation / GCM ([2602.16080](https://arxiv.org/abs/2602.16080)) | 2026 | `papers/generative-causal-mediation-where-to-steer-2602.16080.pdf` | **S1 定位（该 steer 哪里）** |
| C1 | ReFT / LoReFT ([2404.03592](https://arxiv.org/abs/2404.03592)) | NeurIPS 2024 | `papers/reft-representation-finetuning-loreft-2404.03592.pdf` | **B screen 血统** |
| C2 | RePS: Reference-free Preference Steering ([NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eb1ef82926376d252dde00d5dd909f4b-Abstract-Conference.html)) | NeurIPS 2025 | `papers/reps-reference-free-preference-steering-neurips2025.pdf` | **S3 训练目标** |
| D1 | From Weights to Activations（RepE 综述/分类学） ([2026.acl-long.1377](https://aclanthology.org/2026.acl-long.1377/)) | ACL 2026 | `papers/from-weights-to-activations-repe-survey-acl2026-long1377.pdf` | 方法排序 + 功能坐标 |
| D2 | Conditional Activation Steering / CAST ([2409.05907](https://arxiv.org/abs/2409.05907)) | ICLR 2025 | `papers/conditional-activation-steering-cast-2409.05907.pdf` | **S3 门控同构** |

## 文档结构

- **[01_STEERING_LITERATURE_DEEP_READ.md](01_STEERING_LITERATURE_DEEP_READ.md)** — 九篇逐篇深读：核心机制 + 关键量化结论。
- **[02_VZ_IMPLICATIONS.md](02_VZ_IMPLICATIONS.md)** — 落到 Volvence：S2 null 的文献归因、可直接跑的 steerability 预检、S2'/S3 的配方与 B screen 血统对齐。
- **[03_STEERABILITY_PRECHECK.md](03_STEERABILITY_PRECHECK.md)**（P1，只读诊断）— 用 [`scripts/run_steerability_precheck.py`](../../scripts/run_steerability_precheck.py) 实测：probe d′ 4.4–8.6（强解码）vs diff-of-means d′ 0.34–0.90（弱可分）+ 两轴近正交，钉死"可解码≠可静态 steer"、给出 S2 null 的几何成因。
- **[04_HEADROOM_AUDIT.md](04_HEADROOM_AUDIT.md)**（P2a，只读诊断）— 当前 rate-distortion 仪器对"条件/切换干预"无可测余量（恒定算子已捕获 98% distortion、`permuted_z_penalty=0`、静态增益 +2e-4）⇒ 需重设计冲突映射仪器。
- **[05_CONDITIONAL_STEERING_PREREG_SKELETON.md](05_CONDITIONAL_STEERING_PREREG_SKELETON.md)**（P2b，设计冻结骨架）— 冲突映射仪器 + 冻结 S1 condition 传感器 + rank-8 ReFT 执行器 + 显式门控 + instrument-validity 门与 boundary-gated>always-on>random-gate>noop 的 matched-budget 判定门。
- **[06_CONFLICT_INSTRUMENT_VALIDITY.md](06_CONFLICT_INSTRUMENT_VALIDITY.md)**（P2c · C1，已执行只读）— 目标剥离路口仪器 **VALID**：恒定算子错误 0.461、(view,subgoal) 残余歧义 0、基底 goal-stripped NLL 2.81 vs revealed 0.22 ⇒ **2.60 NLL 可 steer 余量、因果归属 subgoal**、基底不确定占比 0.81。证明 P2a"无余量"是 V4 仪器缺陷而非死路，达成 05 的 instrument-validity 前置门，解锁 C2。owner 模块 [`eta_conflict_instrument.py`](../../packages/vz-runtime/src/volvence_zero/agent/eta_conflict_instrument.py)。
- **[07_CONDITIONAL_STEERING_SCREEN.md](07_CONDITIONAL_STEERING_SCREEN.md)**（P2c · C2，已执行）= **PASS** — rank-8、no free bias、zero-code no-op 的学习式乘性写入，按 subgoal 条件化后把 heldout expert NLL 从 2.81 关到 **0.027**（＜文本天花板 0.22）；等预算 **unconditional 只到 1.36**（条件优势 1.33），**random-condition 7.38**（错条件反伤）。3 seed 全过 4 门 + 结构门。**证明"读残差 + 有界学习式执行器 + 按 subgoal 条件出手"能 steer 且条件性有独立因果价值**；不复活 `kill-eta`。owner 模块 [`eta_conditional_steering_screen.py`](../../packages/vz-runtime/src/volvence_zero/agent/eta_conditional_steering_screen.py)。
- **[08_READ_STEER_S3_PREREQ.md](08_READ_STEER_S3_PREREQ.md)**（P2c · S3 前置，已执行）= **PASS** — 把 C2 的 condition 从 **oracle** 换成**在线非 oracle sensor**：cheap 审计发现现成 S1 v2 probe 不迁移到 C2 面（top-1 0.145≈chance），但在**携带目标的上下文残差**上 refit 一个冻结线性 reader 可把 subgoal 读到 heldout **1.000**（剥离面 0.164≈chance，符合"条件须来自上下文"）。用读出的条件驱动 C2 执行器扳目标剥离动作：`conditional-online` NLL **0.023 = 完全等于 oracle**，比等预算 unconditional（1.39）优 **1.37**，route-level bootstrap 95%CI 下界（最差 seed）2.40 / 1.26 均 >0，5 seed 全过。**读得到 + 扳得动 + 条件有价值三层就绪，S3 Internal RL 可准入**；`production_promotion_authorized=false`。owner 模块 [`eta_read_steer_prereq.py`](../../packages/vz-runtime/src/volvence_zero/agent/eta_read_steer_prereq.py)。
- **[09_S3_INTERNAL_RL_PREREG_SKELETON.md](09_S3_INTERNAL_RL_PREREG_SKELETON.md)**（S3 本体，设计冻结，**已 S3-A 背书**）— 前置证了"读得到 + 扳得动"，S3 只做第三层：**用 PE/结局信用在线学"何时扳"**。冻结 reader（sensor）+ 冻结 rank-8 executor，唯一在线更新的是门控策略 π(a_t\|PE 代理)，动作 `{noop, steer(s)}`。**复用边界（修正）**：`CausalZPolicy` 耦合 ETA z 空间**不可复用**，只复用 `sparse_proof_reward_taxonomy`/delayed credit 的信用契约语义，策略本体 = owner 模块内自写最小 REINFORCE+baseline。正式判定门冻结于 prereg JSON（SHA `62454418…`）。
- **[10_GATING_HEADROOM_AUDIT.md](10_GATING_HEADROOM_AUDIT.md)**（S3-A · 门控余量审计，已执行只读）= **PASS**。用诚实的**过期 belief**（记忆滞后，切换路口错条件）制造余量：post-switch 子集 always-on **4.16 > noop 2.53**（错条件净损），择时把整体从 always-on 1.79 降到 oracle **1.09**（余量 0.70、vs noop 增益 1.72）；staleness **完全可检测**（P(belief≠fresh\|post-switch)=1.0，误报 0），可观测 pe-hard 门完全等于 oracle 门 ⇒ 策略有干净信号可学。准入 S3-C。
- **[11_S3_INTERNAL_RL_RESULT.md](11_S3_INTERNAL_RL_RESULT.md)**（S3 本体，已执行）= **实质学习性已证；预注册 worst-seed 稳健门未过**。冻结 sensor+executor，唯一在线更新的门控策略只观测 PE 代理、只拿每-episode 终局稀疏信用 `R=-mean(route NLL)`、从不给每步标签，自写 minibatch REINFORCE+advantage 归一化+熵正则。**5 seed 中 4 个稳健学出 selective gate**（pe_gated 0.61–0.92 ≪ always-on 1.79、selectivity 0.35–0.56、CI 强正、优于 oracle 1.09），seed 平均 pe_gated **0.951** 胜所有基线；**1 个 seed 探索塌缩到 always-steer**（selectivity 0）。预注册要求 worst-seed CI>0 ⇒ literal admission **FAIL**；实质证据支撑"稀疏信用可学何时扳"。稳健化（多重启/熵退火）是唯一剩余缺口。`substrate_trainable=0`、reader/executor 冻结、production 未提升。

## 一句话结论

> Steering 是真实的适应范式，但"**拿 probe 权重当方向、在饱和位置单点静态加一把**"恰好是文献里被反复定罪的最弱变体。
> 我们的 S2 null 与学界的可靠性负结果同构；出路是**先做 steerability 预检（方向一致性 + 可分性），把方向来源换成 diff-of-means / 学习式（CAA→ReFT），在有余量处测，用 CAST/RePS 做条件化与策略学习**。
