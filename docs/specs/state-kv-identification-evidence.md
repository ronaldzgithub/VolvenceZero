# 状态载体识别证据（Carrier-Identification Evidence）

> Status: P0-contract 与 P0-smoke 已落地（runner + `verdict_identification.json` 可复跑）；真实 substrate 与盲裁判（P1-directional）pending
> Last updated: 2026-07-26
> 对应需求: R4（token 空间之上的内部控制）、R8（快照优先）、R11（内部状态可命名可发布）、R12（评估只读）、R15（可解释 + 可回滚证据）
> 上游 spec: [`personal-conditioning.md`](./personal-conditioning.md)（16 维状态与三臂投递形态的 owner）、
> [`continuum-memory.md`](./continuum-memory.md)（`mb.*` 行为证据阶梯）、
> [`companion-ablation.md`](./companion-ablation.md)（同基底纪律与跨家族裁判纪律的先例）

## 要解决的问题

对外主张「关系与记忆不是 prompt 工程，而是在模型层被解决」时，唯一有说服力的证据形态不是分数更高，
而是**可被第三方逐字节复核的载体隔离**：同一句输入、同一份 system prompt 字节、空上下文窗口，
不同的人得到不同且可被认回的回答。

本 spec 定义这条证据的实验设计、载体清单、判据与反主张边界。

**先承认现状**：默认路径做不到这个实验。
[`prompts.py`](../../packages/vz-runtime/src/volvence_zero/agent/prompts.py) 会把
`prompt_residue_summary`、`speech_plan`、regime guidance、boundary 标签等**状态派生文本**拼进
system prompt，而 `prompt_residue_summary` 在
[`runtime_helpers.py`](../../packages/vz-application/src/volvence_zero/application/runtime_helpers.py)
里直接包含检索到的记忆条目原文（`"Carry forward continuity from prior context: " + content`）。
即：**默认路径的记忆是 prompt 载运的**。任何人 diff 两次调用的 system prompt 就能看到差异。
因此本证据必须跑在专门的「纯潜通道臂」上，而不是默认臂；把默认臂说成 prompt-free 是伪证。

## 核心实验：同句双人识别（Twin-Session Identification）

一句话：**两个用户各有 N 次会话历史，第 N+1 次会话都只说同一句话，system prompt 逐字节相同、
上下文窗口为空，两个回答却能被盲裁判以显著高于随机的准确率认回各自的人。**

### 素材与前置

| 项 | 来源 |
|---|---|
| 多会话历史与 per-user 持久化、跨用户隔离 | [`tests/longitudinal/test_cross_session_learned_state_continuity.py`](../../tests/longitudinal/test_cross_session_learned_state_continuity.py)（20 session harness） |
| 零上下文 | `ResponseContext.prior_turns` 默认空元组；跨会话状态只能来自 owner hydration，不来自上下文窗口 |
| 状态载体 | `personal_conditioning` owner 的 16 维快照 → residual hook（[`personal-conditioning.md`](./personal-conditioning.md) §3） |
| 裁判纪律 | substrate 为 Qwen 时裁判必须跨家族（沿用 companion-ablation #71/#72） |

### 载体清单（Carrier Inventory）

实验成立与否取决于「状态 → 输出」的每条通道是开还是关。四条臂的通道状态必须逐条声明，
不声明的通道等于未受控。

| # | 通道 | 纯净臂状态 | 说明 |
|---|---|---|---|
| C1 | system prompt 文本 | **关**（字节相同，哈希公开） | `prompt_state_delivery="suppressed"`，只保留不变表达规则段 |
| C2 | 上下文窗口 / `prior_turns` | **关**（空） | 无历史、无检索原文、无摘要 |
| C3 | residual 条件化偏置 | **开** | 被验证载体（16 维 → hook 层常量加性偏置） |
| C4 | temporal 控制码 `z_t`（`control_parameters` / `control_scale`） | **开** | 同属模型层载体，需可独立消融 |
| C5 | 解码配置（profile / temperature / max tokens / 约束后处理） | **报告，不默认关** | 采样层通道，不是模型层；必须进 attestation 供复核 |
| C6 | 权重 / adapter | 关（P0/P1） | 情节级记忆沉淀属后续阶段，见 §反主张 |

C5 是最容易被攻破的一条：解码配置由 assembly 派生，因此**同一臂内两个用户的解码配置可能不同**。
本 spec 不假装它已关闭，而是要求把它算进 attestation 并作为强弱主张的分档条件（§判据 第 4 条）。

### 四臂矩阵

| 臂 profile | conditioning wiring / mode | prompt 状态段 | prompt 哈希 | 预期 matching |
|---|---|---|---|---|
| `state-kv-arm-a-pure` | SHADOW（关） | suppressed | 与 E-pure **相同** | ≈ 随机 |
| `state-kv-arm-e-pure` | ACTIVE / residual | suppressed | 与 A-pure **相同** | 显著 > 随机 |
| `state-kv-arm-bprime` | ACTIVE / text | text（含渲染状态段） | 与 A/E **不同** | 显著 > 随机 |
| `state-kv-arm-e` | ACTIVE / residual | text（现状） | 与 A/E-pure 不同 | 参考，不用于本判据 |

`bprime` 不是拖后腿，它是**信息量对照**：它与 `e-pure` 消费同一份 typed readout
（同一 `rendered_statement` 的 SSOT），因此「E-pure 赢是因为偷偷多喂了信息」这条反驳被结构性排除。
`state-kv-arm-e` 保留为与既有 P0 证据的连线，但因 C1 未关，不进本判据。

## 判据（四条，缺一不可）

1. **`claim_prompt_identity`** — A-pure 与 E-pure 的 `prompt_fp`（system prompt + messages 的
   canonical SHA-256）在每一个 turn 上相等，且 `prompt_state_sections == 0`。
   不等即 `fail`：整个实验作废，不进入后续三条。
2. **`claim_output_divergence`** — 同臂内两个用户的输出确实分叉（token-level divergence > 0，
   且 residual runtime 上报 `personal_conditioning_applied=True`）。
   trace-only synthetic runtime 上报 `False` 时必须记为 `insufficient_data`，不得当作注入成功。
3. **`claim_identification_above_chance`** — 盲裁判二选一 matching 的 bootstrap CI 下界 > 0.5。
   裁判只看回答 + 两份候选用户历史摘要，不看臂标签、不看 prompt、不看任何内部状态。
4. **`claim_carrier_causality`** — A-pure 塌回随机（CI 覆盖 0.5）**且** `bprime` 显著 > 随机。
   前半句排除提示词残留与随机分叉，后半句确认同信息在文本载体下同样成立。

**C5 分档**：若全部 turn 的 `decode_fp` 在两个用户间也相等，第 3 条记为 `retain-strict`；
若 `decode_fp` 存在差异，最高只能记 `retain-prompt-closed`，且 verdict 必须显式写明
「采样层通道未关闭」。禁止在 `decode_fp` 不等时对外表述为「唯一差异是模型内部状态」。

整体状态机：`insufficient_data` / `fail`（1 或 4 不成立）/ `weak-positive`（3 正但 CI 触零）/
`retain-prompt-closed` / `retain-strict`。

## 契约

### 新增运行时开关

| 项 | 契约 |
|---|---|
| `FinalRolloutConfig.prompt_state_delivery` | `"text"`（默认，现状行为逐字节不变）/ `"suppressed"` |
| `ResponseContext.prompt_state_delivery` | 同上；由 `session_observation` 从 config 透传 |
| 互斥校验 | `personal_conditioning_mode="text"` + `prompt_state_delivery="suppressed"` 必须构造期 raise——渲染语句会被丢弃，属静默回退（AGENTS §6 禁止） |

### prompt 段落分层（SSOT）

`build_system_prompt` 的段落分为两类，`prompts.py` 是唯一定义处：

- **不变表达规则段**（invariant）：人格与语气基调、只回复最新 user 消息、紧凑自然、语言匹配、
  不暴露内部模块名。这些段不消费 assembly / context 的任何状态字段，因此对全部用户全部会话逐字节相同。
- **状态派生段**（state-derived）：regime guidance、`personal_conditioning_statement`、
  `prompt_residue_summary`、`speech_plan`、citation / clarification / refer-out / disclaimer 标签、
  ordering、regime switch 提示。`suppressed` 下整组不进 prompt。

### 审计标签（rationale tags）

沿用既有 tag 审计通道（与 `personal_conditioning` / `personal_conditioning_text` 同级）：

| tag | 含义 |
|---|---|
| `prompt_fp=<sha256[:16]>` | 实际送入 substrate 的 system prompt + chat messages 的 canonical 指纹 |
| `prompt_state_sections=<n>` | 本轮进入 prompt 的状态派生段数量；`suppressed` 下必须为 0 |
| `decode_fp=<sha256[:16]>` | 解码相关配置（profile / temperature / max tokens / 约束）的指纹 |

三个 tag 在**所有** LLM turn 上无条件发出，不只在实验臂——否则「默认臂也没有 prompt 载运」
这类说法无法被反驳，也无法被证实。

## 关键不变量

1. **`suppressed` 是证据专用，禁止部署**：该模式会移除 boundary / disclaimer / refer-out 的
   prompt 侧引导。边界约束在该臂下仅由 `GenerationConstraints` 的 substrate 侧后处理承担。
   守门测试断言默认 profile 与默认 `FinalRolloutConfig` 一律不使用 `suppressed`。
2. **同 substrate 字节级一致**：四臂命中同一份冻结权重、同 seed、同 scenario；沿用
   companion-ablation 的 `substrate_fingerprint.json` 校验。
3. **evaluation 只读（R12）**：matching 准确率是 readout，不回灌学习链路、不进 reward。
4. **信息量同源**：`bprime` 的文本与 `e-pure` 的向量必须来自同一 `PersonalConditioningSnapshot`
   的两种投递形态，禁止为文本臂另做一份摘要。
5. **裁判跨家族**：substrate 家族不得同时充当 matching 裁判。
6. **哈希覆盖真实入参**：`prompt_fp` 必须在送入 `runtime.generate()` 的同一对象上计算，
   不允许重算一份"应该等价"的 prompt。

## 能证明什么，不能证明什么（反主张）

**能**：关系姿态、情绪负荷、决策准备度这类**压缩 readout 的校准**发生在模型层，不依赖提示词。

**不能**：

- **不能证明情节事实记忆在模型层**。当前 residual 通道是 16 维、单层、常量加性偏置、
  `scale ≤ 0.12`，带宽塞不进「她的猫上周去世」。`personal-conditioning.md` 自身即规定
  「精确事实、原话和证据继续走现有可审计上下文」。情节级主张需要 KV / adapter / 权重载体
  （C6），属 State KV P3 与 dual-track adapter 沉淀，gate 在 GPU bake（debt #41），
  在此之前对外声称即为超发。
- **不能证明 16 维是学出来的表征**。向量由 owner 确定性模板从上游 typed readout 编译。
  「学习增量」的主张由 companion-ablation 的 `claim_training_adds_value`（volvence vs
  volvence-cold）承担，不由本实验承担。
- **不能外推到物理具身**。人类侧成功不构成世界模型主张（沿用 companion-ablation 口径）。

## 执行阶段

| Phase | 内容 | 真钱 | 状态 |
|---|---|---|---|
| P0-contract | prompt 载体开关 + 三个 attestation tag + 两个 pure 臂 + 契约测试 | 0 | ✅ 已落地 |
| P0-smoke | deterministic-fake substrate 跑通四臂，验证 `prompt_fp` 相等与 tag 完整 | 0 | ✅ 已落地，见下 |
| P1-directional | 真冻结 Qwen + 跨家族盲裁判，2 personas × K 探针句 × 多 seed | 中（裁判） | pending |
| P2-retain | held-out personas + 多 seed，出 held-out `verdict_identification.json` | 批准预算 | pending |

### P0-smoke 实测结果（2026-07-26）

`scripts/run_state_kv_identification.py` 跑四臂 × 2 personas × 3 探针句，
产出 [`artifacts/state_kv/verdict_identification.json`](../../artifacts/state_kv/verdict_identification.json)。
runner 与判据实现见 `packages/vz-runtime/src/volvence_zero/state_kv_identification.py`。

| 项 | 结果 |
|---|---|
| `verdict_state` | `insufficient_data`（fake substrate 的诚实上限） |
| 判据 1 `claim_prompt_identity` | **pass** —— 两条 pure 臂 12 个 turn，逐探针共享同一 `prompt_fp`，`prompt_state_sections=0` |
| 判据 2 `claim_output_divergence` | `insufficient_data` —— trace-only runtime 上报 `applied=False`，按 §判据 2 不得当作注入成功 |
| 判据 3 / 4 | `insufficient_data` —— 无盲裁判；runner 不会为缺失的裁判编造 matching 数值 |
| C5 分档 | `decode-divergent` |

两条对后续阶段有约束力的结论：

1. **判据 1 是真的成立的**，不是断言：两个 persona 的 assembly 携带完全不同的
   regime 名、记忆残留原文与 disclaimer，`suppressed` 下这些状态派生段全部不进
   prompt，因此 `prompt_fp` 逐字节相同。这条是 P1-directional 的前置，现在已闭合。
2. **C5 在真实 assembly 下默认是开的**：`GenerationConstraints` 的
   `ordering_bias` / `required_disclaimer_phrases` / `decoding_profile` 由 per-user
   assembly 派生，且被 substrate 侧后处理真实消费。因此即便 P1-directional 的盲裁判
   全部命中，最高只能记 `retain-prompt-closed`，**拿不到 `retain-strict`**——除非
   实验设计额外把两个 persona 的解码配置对齐。这是花裁判预算之前必须先决定的事。
   （`prompt_residue_summary` 也在 constraints 里逐字携带记忆原文，但当前无任何
   substrate 消费该字段，故它今天不是载体；一旦有 consumer 就会在 `suppressed`
   下静默打开一条记忆通道。）

runner 的反超发齿：`SubstrateEvidenceKind.TRACE_ONLY` 下即使四条判据全过，
verdict 也被强制封顶在 `insufficient_data`，并写明「retained verdict 需要
residual hook 真实触发的冻结权重」。因此本次零成本 smoke 不可能被引用为识别结论。

## 接口契约

**消费**：per-turn `AgentResponse.rationale_tags`（三个 attestation tag）、
`PersonalConditioningSnapshot`、per-user 持久化状态、`substrate_fingerprint.json`。

**产出**（包 2）：`verdict_identification.json` —— 四条 claim 状态、C5 分档、
matching 准确率与 CI、prompt 哈希对照表、substrate fingerprint、裁判模型标识。

## 回滚

`prompt_state_delivery` 默认 `"text"`，为现状行为的逐字节等价路径；两个 pure 臂是显式 profile，
不进任何默认矩阵。整包 `git revert` 即回滚，无状态迁移。
