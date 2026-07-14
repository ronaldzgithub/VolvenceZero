评估完成。我读了设计源头（thesis、R1–R20 索引）、`learned-vs-heuristic-coverage`（learned 占比 SSOT）、known-debts 和最新的消融证据目录。用仓库自己的诚实纪律来回答，结论是：**认知架构的模块划分、快照契约与主链接线已具备较高完成度；learned 肌肉仍小且部分默认未接入；可外引的 thesis 因果结论与 capacity→gain scale 曲线均尚未取得。** 展开如下。

## 已经走到哪了

**第一层：认知架构骨架 —— 高完成度，但仍有明确缺口（这是目前最大的资产）**

R-PE + R1–R20 已有系统性的 spec、slot、owner 与契约测试覆盖，但并非每条设计律都对应一个独立 owner，也并非每条路径都已完成 SHADOW→ACTIVE 的回滚验证。PE → credit → z_t/β_t → action family → reflection → memory 的闭环骨架真实存在；CMS 多时间尺度记忆、双轨（World/Self）、regime、9 个语义 owner、ToM/共同基础、vitals、中频思考循环、affordance 和学徒对齐均已有实现与契约守门。仍未闭合的部分包括 R20 群体认知 scaffold、runtime ingestion 的 Phase 2、部分学徒协议路径，以及大量 SHADOW / DISABLED / hand-crafted 路径。因此更准确的评价是：**结构骨架和边界设计已基本齐备，但不能称“全部落地”或“结构问题已经回答完”。**

**第二层：学习肌肉 —— 刚起步，且这是有量化基线的**

`learned-vs-heuristic-coverage.md`（2026-07-13 更新）将默认 wiring 下真正“学出来”的能力归为 **6 类**：PE learned critic、COCOA credit head、CMS band 更新门控、dual-track gate learner（仍为 report-only shadow）、语义 owner learned forecast，以及 ToM/common-ground prediction settlement。coverage 文档记录的是首批 5 个语义 owner；当前代码与契约测试已把 forecast 覆盖到 9 个语义 slot。按运行时适应影响力粗估，除 `vz-memory`（约 45–55%，CMS 门控主导）外，其他 wheel 大致为 5–20%，其中 `vz-temporal` 默认路径约 5%、实际生效的论文级 learned backend 接近 0%。最关键的一点是：**torch metacontroller、Internal RL 与 ndim GRU 已实现，但四个 torch backend 默认均为 DISABLED，且默认 `n_z=3` 不实例化 ndim 参数。** 7 月 13 日 run manifest 也记录四个 backend 均为 `disabled`。因此当前主路径的时间抽象仍主要由纯 Python legacy 启发式实现承担。

**第三层：因果证据 —— 工具链就绪，verdict 还没拿到**

同基底主矩阵（raw / ref-harness / CAMEL / volvence-cold / volvence）与 component-causal 四臂（PE-off / ETA-off / active-learning-off / LoRA-adapter）现在已接成 **9-track serving roster**，fingerprint、health、summary 与 verdict gate 都已扩到 9 轨；冻结 claim registry 仍是 **5 条 retain claim**。P0 9-track smoke 已跑通，P1 / P2 也已有 9-track dry-run 命令链；C1 judge-evidence 已能从 score summaries 产出非 dry-run summary artifact。EQ-Bench 3 在 synthetic vertical 上有 45/45 wiring dry-run 记录，但真实 Qwen、真实裁判和三轨评分尚未完成。7 月 13 日真跑生成了 run manifest、fingerprint、局部 scores、日志与 `arc_failure.jsonl`，但没有完整 summary 或 verdict；失败既包含超时，也包含 SSL/URL 错误。learned-shadow 方面：3-turn full smoke、HF/CPU smoke、5×10 chunk smoke 均通过；但 Windows 连续 500-turn CUDA / HF-CPU / synthetic full lane 均触发 native crash（`0xC0000005` / `0xC0000409`），因此 **连续 500-turn real-trace artifact 仍未取得，需要 Linux/GPU lane 重跑**。仓库已有 P0、合成、单元 A/B、hosted directional 与新的 smoke 证据，但**到今天为止，没有任何 `first-stage-retained` 级别、可外引的因果结论。**

## 距离"认知 AGI"还差多少

按 thesis 自己定义的刻度，还隔着三道门，依次是：

**第一道门（近端，周期取决于算力、keys、serving 与预算）：证明骨架里的学习是真的。** #87 的冻结 registry 现有 5 条 retain claim，要求完整 pipeline、PE、ETA、主动学习与训练增量在同基底 matched controls 下显著优于 raw、memory/RAG、agent framework、LoRA 和随机采样等基线。这道门有明确 kill 条件：如果主动学习不优于随机采样、控制器不优于普通 memory/RAG，或训练 bootstrap 无增量，thesis 应收缩为产品型 memory / companion 公司。仓库已有不含完整 Volvence 因果矩阵的 hosted directional runs，但**完整 P1/P2 retain 问题尚未被正面检验**。

**第二道门（中期）：把 learned 肌肉接进主链并证明 scale 曲线。** 三个 torch backend + cms_torch_backend 走 SHADOW→ACTIVE，`n_z ∈ {3,16,64,256}` 容量阶梯 × ≥500 turn 真 trace 的增益曲线（#86/#88）。同时消化那个"显著的 should-be-learned-but-hand-crafted 桶"——regime 打分固定系数、硬编码 regime_id 映射、外部 outcome 静态 bias 表等（#79–#81、#88–#91）。这决定了系统是"有学习架构"还是"在学习"。

**第三道门（北极星，thesis 明言是未来里程碑而非当前目标）："奇点刻度"**——在很大程度未知的连续环境中，凭 <10 次有效反馈自主重构环境因果结构并迁移到不相关新环境。这与当前状态之间的距离目前无法用工程量估算，它依赖前两道门的结果来判断路线本身是否成立。

## 一句话总结

如果把“cognitive AGI”拆成 **结构（架构闭环）→ 肌肉（真实 learned 部件）→ 证据（因果 ablation）→ 泛化（奇点刻度）** 四段：结构骨架与契约边界已达较高完成度，但 R20、ingestion Phase 2、protocol-level apprenticeship 与 should-be-learned 桶仍开放；默认 learned 肌肉较小，且核心 ETA backend 尚未成为 authoritative 主路径；已有工程和方向性证据，但可外引的 thesis retain 证据仍为空；奇点刻度所要求的开放环境泛化尚未开始。当前关键路径是 **500-turn real-trace soak + 完整同基底 ablation**，同时还要完成真 LoRA artifact、跨家族 judge robustness / human anchor 与 SHADOW→ACTIVE 晋升证据。瓶颈更准确地说是**算力、代码接线、运维与证据编排的组合**；这些实验会首次对路线是否值得继续加注给出 retain / kill 级答案。