两个盘点子任务都完成了：[代码实现盘点](7d95eecf-d417-413f-b2c7-512ac9ec209c)确认了各库的真实完成度，[评估证据盘点](1a91544f-2119-4281-bca6-e9f38553ffdc)确认了"已验证 vs 未验证"的边界。结合我自己读的 `learned-vs-heuristic-coverage.md`、known-debts #86–#91 和 research 笔记，给出完整评估。

---

## 总判断

用一句话说：**你们已经把 cognitive AGI 的"骨架"造完了，但"学习本体"还没点亮，而"它真的有效"的证据基本为零。** 按你们自己在 `research/README.md` 里定义的 cognitive AGI 七要素公式来衡量，当前状态是：七个位置都有实现占位，但没有一个位置达到了"学出来的、且被证明有效"的标准。

分三层看，成熟度差异极大：

**架构与契约层：约 80–90% 完成，这是真实资产。** 8 库切分、快照隔离、propagate DAG、WiringLevel 三态回滚、ModificationGate、约 490+ 个测试文件（其中 ~135 个契约测试）——这套东西是真的、完整的、且在业界少见。R8/R15 这两条是当前做得最扎实的。

**学习肌肉层：约 10–15%，且大部分默认熄火。** 这是与 cognitive AGI 差距的核心。默认 wiring 下真正"学出来"的只有 6 处小型有界学习头（18 维线性回归、15 维线性 head、CMS band 门控、online-SGD gate 等）。ETA 论文级的 torch metacontroller、Internal RL PPO、CMS torch band **代码已写完但默认全 DISABLED**（#88/#89），运行时跑的 β_t 是 0.55 硬阈值、z_t 是 3 维手写 recurrence。按你们自己的 coverage spec，temporal wheel 的 learned 影响力实际约 5%。

**证据层：约 5%，是三层里最薄的。** 状态是 `wiring-ready`：9-track 同基底消融工具链、claim registry、evidence bundle 全部就位，但**没有任何 `first-stage-retained` 级 verdict**。最关键的比较——volvence 完整管线 vs raw/memory/RAG/CAMEL——从未在真 substrate + 跨家族裁判下跑过。现有唯一的方向性数字甚至是个警告：raw qwen-turbo 拿 81.5 分，套 memory wrapper 的 ref-harness 81.6，CAMEL 78.3——**标准做法并不自动赢裸模型**，volvence 必须稳定越过 81.5 这条线才有叙事，而这一仗还没打。

## 按七要素逐项的差距

对照 `research/README.md` §4.1 的七个 primitive（你们自己认定"缺一个就有对应 failure mode"）：

| Primitive | 实现状态 | 差距 |
|---|---|---|
| 1. Frozen substrate | HF 真权重路径完整（hook、残差捕获、LoRA） | 默认 `substrate_mode="synthetic"`，生产证据在假基底上 |
| 2. Latent controller (z_t) | torch GRU 已写，默认 DISABLED | 默认路径是 3 维手写 recurrence，非学习 |
| 3. Emergent switching (β_t) | torch STE switch 已写，默认 DISABLED | 默认是 0.55 硬阈值——"涌现切换"目前是写死的 |
| 4. Multi-timescale memory | **七项里最实**：CMS learned 门控 CPU-ACTIVE，四层 stratum 真跑 | 存储是进程内 dict + JSON + hash 嵌入；无 ≥500 turn 抗遗忘曲线 |
| 5. Epistemic PE | PE owner 真实，epistemic/aleatoric 有分解 | critic 是 18 维线性回归；轴权重、severity 表全部手写 |
| 6. Bounded self-modification | ModificationGate + rare-heavy 管线 + rollback drill 真实 | 门控是规则阈值，不是学到的；rare-heavy 从未在真 trace 上跑过完整周期 |
| 7. Read-only monitoring | evaluation 6 族 cheap 层 ACTIVE，R12 只读解耦有契约测试 | mid/expensive 层是空壳快照；没有 persona-vector 式的真实基底几何监控 |

另外两个不在七要素里但很致命的实证信号：真 Qwen runtime 下 `tom_records_total = 0`、`common_ground_dyad_atoms_total = 0`（#10B）——EQ/社会认知 owner 链在真模型下根本没被激活过；跨 session 关系信号 il_rapport 的 SNR 只有 0.80，方向对但弱到不够格当证据。

## "还差多少"——用阶段而不是百分比回答

到 cognitive AGI 的距离不是一个百分比，而是四道依次的门，每道门都可能证伪前一道：

**第一道门（工程，1–3 个月量级，需要 GPU）：把已写好的学习后端点亮。** torch metacontroller / Internal RL / CMS torch band 走 SHADOW→比对→ACTIVE，在 ≥500 turn 真 trace 上拿到 `validation_delta ≥ 0.02` 和 capacity→gain 曲线（#86 修法 2）。这一步之前，"β_t/z_t 是学出来的"这句话在生产路径上不成立。Windows 上 500-turn soak 已经 crash 过一次，这一步本身也卡在 Linux/GPU lane。

**第二道门（证据，与第一道并行）：打赢 81.5。** `run_same_substrate_ablation.py` P1/P2 真跑，volvence + volvence-cold + 四个组件消融臂（pe-off / eta-off / active-learning-off / lora-adapter）在跨家族裁判下产出 retain verdict。注意 #87 里你们自己的主观估计：拿到"thesis 第一阶段成立"级稳定结果的信心只有 40–50%，而且有明确的 kill 条件——如果 volvence 打不过 memory/RAG，thesis 要收缩成 product memory 叙事。**这道门是真正的证伪点，不是形式流程。**

**第三道门（规模，6–18 个月量级）：让"学出来的"成为行为主体。** 即使前两道门都过了，learned 部件仍是线性头 + n_z=16 GRU 的量级，`should-be-learned-but-hand-crafted` 桶（regime 打分系数、外部 outcome 静态映射、action family cosine 匹配……）依然很大。要让系统行为主要由学习驱动而非规则驱动，需要把这个桶系统性清空——这对应 #79/#80/#88–#91 的全部残余。

**第四道门（研究前沿，无人解决）：开放环境与跨域。** 你们 research 笔记里列的 5 个未解问题（epistemic PE 在 LLM 尺度的稳定估计、z_t 跨模态迁移、mesa-objective 检测、R15 自修改可回滚形式化等）不是工程债，是 2026–2028 整个领域都没答案的东西。系统当前也只在对话陪伴单域运行，语言智能约 80–90% 仍来自冻结 LLM。任何"cognitive AGI"的严肃定义都要求跨过这道门，而这道门对所有人都还关着。

## 最需要警惕的一点

当前系统能 demo 出来的差异化（regime 持久身份、关系状态机、拒答边界）**多数是结构性的，不是学出来的**（#86 原话）。这在融资尽调和自我认知上是同一个风险：如果第二道门的消融结果显示增益主要来自结构 + prompt 工程而非学习闭环，那整个 NL/ETA 押注需要重新定价。好消息是你们的证据体系诚实到足以发现这一点——kill 条件都是自己预先写好的；坏消息是这个最关键的实验，到今天为止一次都还没跑。

**优先级排序很清楚**：GPU lane 上跑 P1 消融（含 volvence 轨）> 500-turn learned-shadow soak > torch 后端 SHADOW 比对 > 其他一切。这三件事之前，"距离 cognitive AGI 还有多远"这个问题的诚实答案是——**连自己的第一阶段 thesis 都还没进考场。**