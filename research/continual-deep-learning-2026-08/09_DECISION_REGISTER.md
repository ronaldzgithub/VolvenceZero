# 09 · 采纳决策登记册

> 本表是研究决策，不修改 runtime、slot、WiringLevel 或 production 默认。任何 `Adapt` 落地仍须单独
> 收敛包、DATA_CONTRACT 注册（如适用）、spec、测试与 promotion evidence。

## 1. 当前决策

| 决策 ID | 对象 | 决策 | 原因 | 下一证据 | 复审触发器 |
|---|---|---|---|---|---|
| D01 | “内部可以控制”作为创新主张 | Reject | J-space/FV/ReFT/TACT 已有强先例 | 改写定位为四轴闭环 | 若出现更严格公开闭环，继续收窄定位 |
| D02 | J-lens-like reader | Adapt | 最强的命名+因果 workspace 仪器 | P0 跨视图/patch 对照 | 当前 open model 无增益或成本失控 |
| D03 | Natural Language Autoencoder | Watch/Proposal-only | 可发现开放词表状态，但文本解释不是 truth | 小规模 proposal validity | causal/calibration 显著胜固定标签 reader |
| D04 | Concept Vector / RSA head selection | Adopt/Adapt | 直接针对 causal-but-format-bound 失败 | P0 reader challenger | 跨视图不胜 v2 reader |
| D05 | direction coherence + separation precheck | Adopt | 低成本、只读、已有内外双重负证据 | 纳入下一 prereg | 新证据显示其无预测力 |
| D06 | ReFT executor | Adapt challenger | 学习控制轴优于拿 probe 轴硬 steer | P1 同 rank/norm 对照 | 不胜当前乘性 executor |
| D07 | pyReFT runtime dependency | Defer | 通用工具有价值，但 contract/后端/依赖未审 | 隔离训练与序列化审计 | 显著性能优势且可收窄依赖 |
| D08 | CAST | Baseline | 最接近条件静态强基线 | G2 固定臂 | learned gate 已稳定胜出后仍保留审计臂 |
| D09 | IBM activation-steering runtime | Baseline tool only | Apache-2.0、易复现；默认 additive/threshold 不符主链 | 隔离复现 | 需要快速重现 CAST 时启用 |
| D10 | TACT drift axes | Adapt method, not vectors | 长轨迹真实终局最接近；轴与标签不可迁移 | coding-lab 轨迹协议 | 两模型复现失败或 outcome 无增益 |
| D11 | per-instance layer scheduler | Measure-first | 新证据强，但扩大 action space 有样本/审计成本 | oracle headroom audit | oracle lift 显著且 frozen ranker 可回收 |
| D12 | adaptive-K dose | Adapt after D11 | 可减少 oversteer/collapse | dose curve | 固定 dose 已足够或 gate 样本不足 |
| D13 | side-effect matrix | Adopt | ACTIVE 的核心缺口，且副作用可结构化 | 12–16 行为首版 | 无预测性也仍保留实测 matrix |
| D14 | side-effect predictor | Watch/Read-only | 有潜力前置风险，但不能当 reward | unsteered-state forecast test | 两模型方向预测过门 |
| D15 | FaithSteer 三门 | Adopt | 防虚假可控、能力税和 perturbation 脆弱 | P6 promotion | 无 |
| D16 | FASB backtracking | Watch | turn-level gate 可能太晚，但 token rollback 改动面大 | 先证明 P3/P4 | 长轨迹中延迟检测成为主因 |
| D17 | PID/LQR controller | Watch | 有稳定性价值，但 setpoint error 不是 PE | offline executor comparison | static executor 出现 overshoot/dose failure |
| D18 | Titans surprise write | Adapt as salience | 与 PE/多频同构；信号语义更窄 | memory write-policy challenger | 不胜现有 gated write |
| D19 | HOPE/CMS 整体架构 | Do not import | 理论血缘近，但系统 owner/治理不同 | 只引用思想 | 无 |
| D20 | TTT-E2E 快权重 | Reject online architecture / Adopt boundary | 违反冻结基底；exact NIAH 是关键反例 | exact-vs-compressed ladder | rare-heavy 独立路线另议 |
| D21 | SEAL self-edit | Rare-heavy reference only | 有学习价值，但慢、写权重、遗忘、外部 reward | 不进入 online-fast | ModificationGate 下独立 artifact 研究 |
| D22 | ETA Internal RL | Adapt policy locus, not reward | latent controller/termination 最接近；仿真外部 reward 不符 | P4 gate 结构对照 | 对话域 PE gate无 headroom |
| D23 | CL-Bench gain + naive ICL | Adopt | 最强持续学习证伪框架 | P5 longitudinal | 无 |
| D24 | Spurious Forgetting diagnostic | Adopt | 防止把调用失败误判知识丢失 | memory failure triage | 无 |
| D25 | vllm-lens | Isolated spike | 机械能力完整、MIT；eager 与 cloudpickle 风险明确 | P6 backend equivalence/SLO/security | typed adapter 不可实现或 overhead 超门 |
| D26 | Goodfire infrastructure | Design reference | frontier scale 可行性；自报非 efficacy | 架构阅读，不进 formal | 出现开源可复现实现 |
| D27 | persona/emotion vectors 作用户状态 | Reject | 表示的是模型概念/局部 operative state，不是用户本体 | 只作 readout candidate | 有长期多源因果验证仍需 owner 审计 |
| D28 | judge/human score 作 gate reward | Reject | 违反 R12；循环自证 | human 仅 validation anchor | 不复审，除非系统原则变更 |
| D29 | universal vector/layer | Reject | 跨模型/格式/样本负证据充分 | artifact 强制 lineage | 不复审为默认假设 |
| D30 | 直接 production ACTIVE | Block | C3/B3、behavioral N+1、SLO/side-effect 未过 | P0–P6 + 正式 promotion | 全部门通过且 ModificationGate allow |

## 2. 近期最高信息价值队列

按“单位成本能消除多少上游不确定性”排序：

1. **P0 readout cross-view + causal patch**：决定现有状态是否真能承担控制；
2. **P1 substrate authority upper bound**：决定是否值得继续 gate；
3. **per-instance oracle headroom**：只读/离线即可判断要不要扩 layer/dose；
4. **12–16 轴 side-effect matrix**：提前发现现有 executor 的能力税；
5. **CAST static baseline**：为 PE-learned gate 建强对照；
6. **vllm-lens isolated spike**：与科学主线并行，但不得进入 production。

## 3. 暂不做的事情

- 不再扩大静态 probe 数量而缺 causal test；
- 不在当前 substrate authority 未过时训练更复杂 RL gate；
- 不把 layer/dose/action space 一次性全扩；
- 不直接下载并嵌入第三方 runtime；
- 不用人格/情绪标签扩充长期 user model；
- 不以更长 context 替代 Appendable 强基线；
- 不以 memory retrieval accuracy 替代跨 session behavior；
- 不用自动 judge 形成闭环 reward；
- 不在 C3/B3 前改变 production 默认。

## 4. 决策复审协议

每次复审只允许三种原因：

1. 新同行评审/一手实证改变了机制边界；
2. Volvence 新 formal 结果通过或证伪某个前提；
3. 目标 backend/model/version 改变，原 artifact 不再适用。

复审记录至少写：旧决策、触发证据、变更理由、新实验/owner、兼容与回滚。不能因实现方便把
`Reject/Block` 静默改成 `Adopt`。
