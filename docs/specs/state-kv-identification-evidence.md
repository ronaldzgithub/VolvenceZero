# 状态载体识别证据（Carrier-Identification Evidence）

> Status: P0-contract / P0-smoke 已落地；P1 residual 两条 artifact 均未过输出分叉门槛，且其唯一分叉经设备交叉验证判定为数值噪声；旧 `teacher-distilled-prefix-v1` 的 P4 机制门定位为门 A fail / 门 B pass，通道退化为 residual 的多层版本；2026-07-28 `teacher-distilled-routed-prefix-v1` 在 `route_weight=1.0` 下通过 P4（Gate A/B pass，`carrier_is_live=true`），但行为识别仍缺跨家族 blind judge；同日新增 `state-strategy-routed-prefix-v1`，用 16 维状态直接生成策略目标。标准 artifact `8064f8b6de8ec215807619f404c84404087109076634d1ffda53112b4684e238` 已在 MPS 上通过 P3 `retain-strict`（embedding judge 12/12，CI 1.0..1.0）、P4 机制门（Gate A/B pass，`carrier_is_live=true`）与 P2 held-out 两组 pairwise 识别（27/32 与 29/32，CI 下界均 > 0.5）。这证明标准 State-KV artifact 已进入系统、被模型读取，并能在未见过的 persona/probe 上保持可识别；默认 ACTIVE 晋升、多 seed 复核和 rollout gate 仍未完成。
> Last updated: 2026-07-28
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
| C6 | 权重 / adapter | 关（P0/P1/P3） | 情节级记忆沉淀属后续阶段，见 §反主张 |
| C7 | Prefix-KV 注意力前缀 | P3 的 `g-prefix-pure` 臂**开**，其余臂关 | 逐层 K/V slot，由 16 维 readout 生成，带宽高于 C3；基底权重不变，artifact 与权重分开发布 |

C5 是最容易被攻破的一条：解码配置由 assembly 派生，因此**同一臂内两个用户的解码配置可能不同**。
本 spec 不假装它已关闭，而是要求把它算进 attestation 并作为强弱主张的分档条件（§判据 第 4 条）。

### 四臂矩阵

| 臂 profile | conditioning wiring / mode | prompt 状态段 | prompt 哈希 | 预期 matching |
|---|---|---|---|---|
| `state-kv-arm-a-pure` | SHADOW（关） | suppressed | 与 E-pure **相同** | ≈ 随机 |
| `state-kv-arm-e-pure` | ACTIVE / residual | suppressed | 与 A-pure **相同** | 显著 > 随机 |
| `state-kv-arm-bprime` | ACTIVE / text | text（含渲染状态段） | 与 A/E **不同** | 显著 > 随机 |
| `state-kv-arm-e` | ACTIVE / residual | text（现状） | 与 A/E-pure 不同 | 参考，不用于本判据 |
| `state-kv-arm-g-prefix-pure` | ACTIVE / prefix_kv | suppressed | 与 A-pure **相同** | 显著 > 随机 |

判据 1/2/3 定义在 `(控制臂, 候选臂)` 这一对上，控制臂恒为 `a-pure`，候选臂由
`candidate_arm_label` 选择：P1 为 `e-pure`（residual 载体），P3 为
`g-prefix-pure`（Prefix-KV 载体）。verdict 里写入 `candidate_arm` 字段——否则两次
claim 状态相同的运行无法区分测的是哪条通道。判据 4 的控制臂与文本臂不随之改变。

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
4. **`claim_carrier_causality`** — A-pure 塌回随机（CI 覆盖 0.5）**且**候选载体显著
   > 随机。P1 residual lane 的候选证据仍由同源信息的 `bprime` 文本臂承担；P3
   Prefix-KV lane 的候选证据由 `g-prefix-pure` 自己承担，避免让文本臂弱点否决
   一条已关闭 prompt 的 prefix 载体。前半句排除提示词残留与随机分叉，后半句确认
   被测候选载体确实携带可识别状态。

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

**判据全部成立时能**：关系姿态、情绪负荷、决策准备度这类**压缩 readout 的校准**
发生在模型层，不依赖提示词。当前 fixed / learned projector 与 Prefix-KV 三条载体
均未过判据 2，所以现有 artifact **尚不能支撑这条对外主张**。

**Prefix-KV 现在能说什么**：只能说「关闭 prompt 载体后，一条有界的逐层 KV 前缀能
在三条探针中的两条上产生跨设备稳定的双人输出差异，而同信息量的 residual 通道
一条都做不到」。这是**带宽层面的比较结论**，不是身份识别结论——错用户负对照
停在 0.508，说明差异尚未被证明携带身份。把它表述成「模型认出了不同的人」即为超发。

**蒸馏路径的结构性天花板**：G 臂的教师就是 B′ 臂，纯蒸馏最多逼近「把状态写进
prompt」的效果（设计方案 §9.3 原文）。即便日后判据 2 通过，可主张的增量也只是
延迟、上下文预算与不可被用户文本覆盖，**不是**超越提示词工程；后者的增量只能
来自 Stage 3 的 PE 闭环。

**不能**：

- **不能证明情节事实记忆在模型层**。当前 residual 通道是 16 维、默认单层（实验 artifact
  可多层）、常量加性偏置、`scale ≤ 0.12`，带宽塞不进「她的猫上周去世」。
  `personal-conditioning.md` 自身即规定
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
| P1-directional | 真冻结 Qwen + 跨家族盲裁判，2 personas × K 探针句 × 多 seed | 中（裁判） | fixed 与 learned projector 均未过判据 2；盲裁判未启用 |
| P3-prefix | 同一冻结 Qwen 加第五臂 `g-prefix-pure`，候选臂换成 Prefix-KV 载体 | 0（本机训练） | p0 / p2 跨设备稳定分叉；p1 同文、负对照随机，判据 2 未过 |
| P4-mechanism | slot 注意力非退化检验（门 A）+ 状态线性可读出探针（门 B） | 0 | 门 A fail / 门 B pass；`carrier_is_live=false` |
| P2-retain-heldout | held-out personas + held-out probes，出 pairwise `verdict_identification.json` | MPS + 本地 embedding judge | ✅ 已落地，见 §P2 |
| P2-retain-seeds | 多 seed / 多 rollout 复核，同一 artifact 的稳定性门 | 批准预算 | pending |

### P1 frozen-Qwen runner

同一入口通过 `--lane` 区分 synthetic smoke 与真实冻结权重：

```bash
python scripts/run_state_kv_identification.py --lane smoke

python scripts/run_state_kv_identification.py \
  --lane p1 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu
```

P1 默认只解析本地 Hugging Face snapshot，缺权重时 fail loudly；只有显式传
`--allow-download` 才允许下载。四臂共享同一个
`TransformersOpenWeightResidualRuntime` 实例，runtime 构造时保持基底冻结，
`temperature=0`，并对两个 persona 使用同一份 `ResponseAssemblySnapshot`。
因此 P0-smoke 故意保留的 per-user C5 差异不会进入 P1；P1 的 `decode_fp` 应在
每个 probe 上一致。若需要采样 / 多 seed，必须在后续盲裁判包中显式实现逐 turn
seed 对齐，当前入口拒绝非零 temperature，避免随机采样冒充模型层载体。

每次运行在 verdict 同目录写三件套：

| 产物 | 内容 |
|---|---|
| `verdict_identification.json` | 四条 claim、C5 分档、prompt/decode attestation |
| `transcript_identification.json` | 四臂实际回答与实际 `rationale_tags`，供盲裁判和现场展示 |
| `substrate_fingerprint.json` | 实际加载 weight 文件列表与内容 SHA-256、runtime origin、冻结标志 |

P1 runner 不内置同家族 judge，也不在缺少 judge 时生成 matching 分数。真实
residual hook 即使让判据 1/2 成立，判据 3/4 与 overall verdict 仍会保持
`insufficient_data`，直到跨家族盲裁判通过正式 seam 接入。

### 首次 frozen-Qwen 实测（2026-07-26）

命令：

```bash
python scripts/run_state_kv_identification.py \
  --lane p1 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu \
  --max-new-tokens 32 \
  --personal-conditioning-scale 0.12
```

实际冻结权重指纹为
`Qwen/Qwen2.5-0.5B-Instruct@857fff1d6ea77f33`；完整 SHA-256 与文件清单见
`artifacts/state_kv/p1/substrate_fingerprint.json`。

| 项 | 结果 |
|---|---|
| `claim_prompt_identity` | **pass**：两条 pure 臂 12 turns，逐 probe 相同 `prompt_fp`，state sections 为 0 |
| residual hook | **真实触发**：E-pure 6/6 turns 上报 `personal_conditioning_applied` |
| `claim_output_divergence` | **fail**：使用语义一致的反事实 persona 向量且 scale 已达契约硬上限 0.12，p0 / p2 两个 probe 的双人输出仍完全相同；仅 p1 分叉 |
| C5 | **decode-matched**：每个 probe 的双人 `decode_fp` 相同 |
| claim 3 / 4 | `insufficient_data`：没有跨家族盲裁判，未编造 matching 数值 |
| overall | `insufficient_data` |

这个结果是固定投影路径的 kill signal，而不是 runner 失败：载体隔离与真实注入均已
闭合；两个 persona 也不再是矛盾的全维同升/同降，而是在 trust / repair /
decision-readiness / autonomy-risk 等坐标上形成有语义的反事实。即便如此，当前
确定性 sine/cosine basis 仍没有把 16 维关系坐标稳定映射成可识别行为。
下一收敛包必须保持冻结基底与同一 `PersonalConditioningSnapshot` 契约，只替换
substrate owner 内的 projector 为可训练、可版本化、可回滚 artifact；在新的
matched ablation 过判据 2 之前，不应花费跨家族盲裁判预算。

#### 量化归因：是幅度不够，不只是方向不对（MPS 复跑 + 直接测量）

同一入口在 `--device mps`、`--max-new-tokens 64` 下复跑，判据 2 更彻底地不成立：
**p0 / p1 / p2 三个 probe 全部双人同文**。且查 `transcript_identification.json`
可见更强的一条事实——E-pure 的输出与**完全没有条件化**的 A-pure 逐字节相同
（`'别灰心，下次再接再厉。'`）：残差在 argmax 路径上是彻底的 no-op，而不是
"改了但改得不像人"。

对照之下 **B′（文本载体）在同一次运行里是有效的**：persona-a `'我明白你的困扰。'`
与 persona-b `'嗯，我理解你现在的状态。不过别担心，每一次失败都是成长的机会…'`
明显分叉。同一份 typed readout 的两种投递形态，一种带得动、一种带不动——
因此瓶颈不在 16 维 readout 的信息量，而在残差通道本身。

直接测量注入幅度（Qwen2.5-0.5B，hidden=896，注入层为 `layer_indices[0]`=11 / 共 24 层；
`build_personal_conditioning_delta` 与 `build_personal_conditioning_basis` 为纯函数，
可独立复算）：

| 量 | 值 |
|---|---|
| `‖Δ_a‖₂`（fill 0.82，scale 0.08，confidence 0.72） | 0.0527 |
| `‖Δ_b‖₂`（fill 0.24，同上） | 0.0188 |
| **两人注入差 `‖Δ_a − Δ_b‖₂`** | **0.0340** |
| 注入层残差流 per-token L2（末 token / 全序列均值） | 13.61 / 93.99 |
| **相对扰动 `‖Δ_a−Δ_b‖ / ‖h_last‖`** | **2.5 × 10⁻³** |
| 同上，取到硬上限 `scale=0.12` | 3.7 × 10⁻³ |

即：两个用户的隐状态在 24 层中的第 11 层相差约 **0.25%**，随后还要穿过 13 层。
`clamp_personal_conditioning_scale` 把 scale 硬钉在 `[0, 0.12]`（personal-conditioning.md
的契约上限），所以这条通道的幅度上限是**结构性**的，任何配置都抬不上去。
作为对照，activation steering 类方法通常使用与激活范数同量级的扰动，比这里高约两个数量级。

**对下一收敛包的约束**：可训练 projector 只能改**方向**，改不了幅度——它仍然被同一个
0.12 cap 约束在 ~0.3% 的扰动天花板下。因此该包必须同时给出幅度侧的处置，三条路各有代价，
需要显式选一条而不是默认继续：

1. **重审 0.12 cap**：cap 存在的理由是有界性与安全性，抬升需要重新论证边界，
   属契约变更（personal-conditioning.md 与 DATA_CONTRACT 同步）；
2. **多层注入**：当前只有 `layer_indices[0]` 收到 personal delta（见
   `residual_backend` hook 中的 `layer_index == self._layer_indices[0]`），
   摊到多层可在不动单层 cap 的前提下提高总影响；
3. **直接进 P3 prefix-KV**：逐层 Prefix-KV 的带宽本就高一个层级，
   与其把固定投影修到勉强可测，不如承认 P2 的价值是"跑通训练管线"
   （设计方案 §14 P2 原文即如此定位），把识别主张压到 P3。

在选定其中一条并让判据 2 在 matched ablation 下成立之前，跨家族盲裁判预算仍应保持为 0。

#### Contrastive projector matched ablation（2026-07-26）

本收敛包选择上面的第 2 条：保持冻结基底和单层 `scale ≤ 0.12` 不变，只在
`vz-substrate` 增加版本化 projector artifact，并把同一 delta 施加到三个 middle
hook 层。方向来自同一冻结 Qwen 的 16 组正/负语义 anchor 残差差，逐行 L2
归一化；artifact 不保存用户事实、不修改基底权重，省略参数即回滚到 fixed 单层路径。

```bash
python scripts/bake_state_kv_projector.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu

python scripts/run_state_kv_identification.py \
  --lane p1 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu \
  --max-new-tokens 32 \
  --personal-conditioning-scale 0.12 \
  --projector-artifact \
  artifacts/state_kv/projectors/qwen2.5-0.5b-contrastive.json
```

artifact ID 为
`36bd71eb317e23ad828bd0fad75109debc1c10ef0d99d16e4dfd5f6b6ecf2e6c`。
manifest 记录相同权重 SHA-256、anchor SHA-256、32 条训练素材、layers
11 / 12 / 13 与 `base_model_mutated=false`。结果写入
`artifacts/state_kv/p1-learned/`：

| 项 | fixed 单层 | contrastive 三层 |
|---|---|---|
| `claim_prompt_identity` | pass | pass |
| residual hook | E-pure 6/6 turns | E-pure 6/6 turns |
| `claim_output_divergence` | fail：p0 / p2 同文 | **fail：仍为 p0 / p2 同文** |
| C5 | decode-matched | decode-matched |
| overall | insufficient_data | insufficient_data |

learned 路径仅在 p1 产生细微措辞差异；p0 仍同时回答“别灰心，下次再接再厉。”，
p2 仍同时回答“是的，你认为继续下去。”。结论是：

1. `contrastive-residual-v1` 保留为可复现的否证 artifact，不晋升为默认 runtime；
2. 盲裁判继续不接线，因为逐字相同的回答不可能产生高于随机的身份识别；
3. 下一包转向带宽更高的 Prefix-KV owner/contract，不再调 anchor、偷偷抬高 cap
   或用 prompt 文本补洞。

### P3：Prefix-KV 载体（2026-07-26）

本包冻结 `state-kv-prefix.v1` artifact 契约（见
[`personal-conditioning.md`](./personal-conditioning.md) §3.4），把候选臂换成
`state-kv-arm-g-prefix-pure`，并在同一份冻结权重
`Qwen/Qwen2.5-0.5B-Instruct@857fff1d6ea77f33` 上一次跑完五臂。

历史 `teacher-distilled-prefix-v1` 生成器由**教师蒸馏**训练：教师是 B′ 臂（同一冻结
Qwen 读渲染状态语句），学生是 G 臂（prompt 无任何状态段，只有前缀）。基底冻结，
只训 122,948 个生成器参数。
素材 16 个状态 × 8 条训练探针 = 128 对；**三条 hold-out 纪律**由测试守门：

1. 训练探针与三条评测探针完全不相交；
2. 训练状态采样限制在两个评测 persona 之间的 `|u| ≤ 0.8` 包络内（第二个因子对主轴
   做了 Gram–Schmidt 正交化，否则采样点会投影越过 `|u| = 1`，即落到评测 persona 上）；
3. 错用户方向既进损失（counterfactual margin），也在训练后单独测量。

```bash
python scripts/train_state_kv_prefix.py --device mps \
  --states 16 --epochs 3 --slots 4 --rank 4 --norm-cap 0.2 \
  --route-weight 0.0

python scripts/run_state_kv_identification.py \
  --lane p3 --model-id Qwen/Qwen2.5-0.5B-Instruct --device cpu \
  --max-new-tokens 32 --personal-conditioning-scale 0.12 \
  --prefix-kv-artifact artifacts/state_kv/projectors/qwen2.5-0.5b-prefix.json
```

artifact ID `1b133cf27e27ca2f27bbbd45f29928881b54a55752b4ffab0f5dbe39308134a6`，
4 个 slot、秩 4、`norm_cap=0.2`。证据写入 `artifacts/state_kv/p3/`（CPU）与
`artifacts/state_kv/p3-mps/`（MPS 复跑）。

| 项 | 结果 |
|---|---|
| `claim_prompt_identity` | **pass**：a-pure 与 g-prefix-pure 12 turns 逐 probe 同 `prompt_fp`，state sections 为 0 |
| 前缀注入 | **真实触发**：G 臂 6/6 turns 上报 `personal_conditioning_applied` |
| `claim_output_divergence` | **fail**：p0 / p2 双人分叉，p1 仍逐字节同文 |
| C5 | **decode-matched** |
| claim 3 / 4 | `insufficient_data`：盲裁判仍未接线 |
| overall | `insufficient_data` |

#### 设备交叉验证是本次最有判别力的一步

同一 artifact 在 CPU 与 MPS 各跑一遍，逐臂逐 probe 记录「双人是否分叉」：

| 臂 | CPU | MPS |
|---|---|---|
| `a-pure`（对照） | 无 | 无 |
| `e-pure`（residual） | 仅 p1 | **无** |
| `bprime`（文本） | p1, p2 | p1, p2 |
| `g-prefix-pure`（Prefix-KV） | **p0, p2** | **p0, p2** |

两条结论：

1. **residual 载体此前唯一那处分叉是数值噪声。** 上一包记录的「learned projector
   仅让 p1 出现细微措辞差异」在换到 MPS 后完全消失。因此该通道不是「弱但真」，
   而是在这个基底上没有可复现的效应——这加强而不是削弱上一包的否证结论。
2. **Prefix-KV 是第一条分叉模式跨设备稳定的潜通道**，稳定性与文本臂 B′ 同级。
   p0 尤其关键：A-pure、E-pure、B′ 三条臂在 p0 都是双人同文，只有 G 分叉
   （CPU：`'嗯，听起来你今天有点不顺心。'` vs
   `'我明白你可能感到沮丧和挫败，但我可以理解你的感受。'`）。

分叉的**归因是干净的**：两个 persona 在 G 臂下拿到同形状、同 slot 数、同 norm cap
的前缀，唯一差异是 16 维状态向量本身。所以臂内双人差异只能来自状态内容，不能
归给「挂了前缀」这件事本身（后者只会同等影响两人）。

#### 但仍然不足以进盲裁判

**错用户负对照停在随机**：训练后测量「学生用自己状态的前缀，是否比用别人状态的
前缀更偏好自己教师的续写」，结果 65/128 = **0.508**。也就是说前缀足以扰动 argmax
决策，却没有可测的身份选择性。蒸馏损失逐轮均值为 1.72 → 1.65 → 1.61，第二轮起
基本走平；配合这个对照，最合理的解读是：生成器主要学到了一个**与状态无关的
通用前缀**，状态依赖部分很弱。

因此：

1. `teacher-distilled-prefix-v1` 保留为可复现 artifact，不晋升默认 ACTIVE；
2. 盲裁判继续不接线——p1 逐字节同文且负对照在随机水平，接上只会买一份 0.5；
3. 判据 2 未过，对外仍不得声称模型层身份识别成立。

**下一步的真实约束**（按代价排序）：负对照在随机水平是比 p1 同文更根本的问题，
先解决状态选择性再谈探针覆盖。2026-07-27 的修法不是加大 slot 数、秩或抬
`norm_cap`，而是在 trainer 中加入 deterministic state→slot route target：同一
16 维状态对应一个固定 slot 分布，并对 prefill 末位真实 attention 的 prefix slot
分布做 cross-entropy。这样训练压力直接落在 key / attention 路由上，避免纯蒸馏
继续收敛成 `w · V(state)` 的常量偏置。重新 bake 前，旧 artifact 仍按本节结论
处理；重新 bake 后必须先跑 P4，再决定是否值得进入 P3/P2。

### P4：机制门（门 A 注意力 / 门 B 可读出）（2026-07-27）

P3 之后卡住的不是探针覆盖，是**错用户负对照停在 0.508**。而这个数字把两种完全
不同的失败混在一起：注意力根本没读那些 slot，还是读了但里面没有状态。这两者的
修法不同，所以本包用两个比识别判据更弱、但能证伪、且不花裁判预算的机制门把它们
分开。

产物 `artifacts/state_kv/p4-diagnostics/`，runner：

```bash
python scripts/run_state_kv_carrier_diagnostics.py --device cpu \
  --train-states 96 --eval-states 32 --shuffle-draws 5 \
  --prefix-kv-artifact artifacts/state_kv/projectors/qwen2.5-0.5b-prefix.json
```

#### 两个门的定义，以及两处必须记录的判据修正

**门 A `claim_slot_attention_read`**：A2 跨 slot 区分度（块内 max/min）在**至少
半数层**超过同范数随机前缀对照；A3 注意力剖面的跨状态离散度 > 跨探针句离散度。

> **A1 被废弃。** 原本还打算要求"slot 注意力质量 > 均匀期望 `S/(S+N)`"。实测
> 否掉了它：零内容前缀拿到 **0.347**、随机前缀 **0.339**，均匀期望只有 **0.16**。
> 近零 attention logit 会吸走真实 token（logit 多为负）不争抢的质量，这是
> attention sink，与内容无关。**任何定义在 slot 注意力总量上的门，一个零张量就能
> 通过**。因此总量只报告、不判定。禁止对外引用"状态 slot 拿到 XX% 注意力"。

**门 B `claim_state_linearly_readable`**：探针打在 **prefill 后真实 prompt 末位的
hidden state** 上——不是打在前缀 KV 上，后者是 readout 的确定性函数，probe 它是
循环论证。ridge 的 alpha 由**训练状态内部按状态分组的 4 折 CV** 在固定网格上选出，
全程不看 held-out。

> **B2 第一版是错的。** 原本要求"打乱标签对照 R² ≤ 0"。但报告的统计量是**跨 24 层
> 取最大值**，而有限样本 R² 零分布虽中心近 0 却有方差，取 24 个的最大值必然为正。
> 这个条件在载体完全正常时也不可能满足——第一次全量跑出的 `fail: leakage` 就是
> 这么来的，与载体无关。修正为：held-out 最优值必须**超出零分布天花板一个 floor
> 的余量**，零分布由 5 次独立打乱抽样取上界。中途还提过一个"低方差坐标主导未加权
> 平均"的假设，验证后**被证伪**（16 个坐标方差 0.004–0.063，无近常数坐标），故未
> 采纳；记在这里是因为按错误假设改聚合方式会改出一个好看但没有依据的结果。

#### 结果

| 门 | 状态 | 数字 |
|---|---|---|
| A（注意力被读） | **fail** | 跨 slot 区分度仅 5/24 层超过随机对照（需 >12）；跨状态离散度 **0.00032** vs 跨句 **0.01852** |
| B（线性可读出） | **pass** | layer 5 held-out mean R² **0.8576**（floor 0.1）；打乱零分布上界 0.1059；无前缀对照 **−0.0521** |
| overall | `carrier_is_live=false` | |

无前缀对照为负是结构性的：pure 臂 prompt 逐字节相同，不挂前缀时不同状态的 hidden
state **完全一致**，探针只能预测均值。runner 逐句实测这一点，不成立即记
`insufficient_data`。

#### 定位结论：状态走的是 value，不是注意力

两个门的组合给出一个此前拿不到的机制事实：**状态确实进入了 prefill 并写进了真实
token 的残差流（R² 0.858），但注意力权重几乎不随状态变化**（跨状态离散度比跨句低
58 倍）。也就是说注意力权重近乎恒定、value 随状态变，输出贡献是 `w · V(state)`
——数学形式上就是一个**恒定增益的状态相关偏置**。

那正是 residual 载体的形式。当前前缀生成器**退化成了 residual 通道的多层版本**：
比单层强（所以 P3 的 p0/p2 能分叉），但完全没有用上注意力"按人路由"的能力，因此
产生不了身份选择性（错用户负对照 0.508）。

这直接指定了下一步该改什么，也排除了两条看起来合理但没用的路：加 slot 数、抬
`norm_cap`——它们只会把这个偏置放大，不会让它变成路由。要改的是让**注意力权重
本身**成为状态的函数：当前低秩生成器把 K 和 V 从同一个 `tanh(Es+b)` 瓶颈线性展开，
K 的状态依赖在 per-head norm cap 之后被压到 softmax 分辨不出的量级。

#### 2026-07-27 Gate A 修复包（代码已落地，artifact 待 rebake）

根因属于 `vz-substrate` prefix artifact trainer：旧训练目标只看 B′ 教师续写的
logprob 与 wrong-user margin，没有任何项要求不同 state 改变 slot attention。
因此 generator 可以主要改 value，让 attention 近似恒定，形成 `w · V(state)`。

修复：`scripts/train_state_kv_prefix.py` 默认启用 routed-attention objective：

- `route_target = softmax(fixed_anchor @ (2 * state - 1) / temperature)`，anchor
  是 deterministic 数值基，不携带语义标签；
- 学生用 prefix-KV 过同一 suppressed prompt，在 prefill 末位读取真实
  `outputs.attentions`；
- 对每层 prefix slot attention 分布与 `route_target` 做 cross-entropy，并与原
  B′ distillation / wrong-user margin 一起优化；
- 新 artifact 默认标记为 `teacher-distilled-routed-prefix-v1`，旧
  `teacher-distilled-prefix-v1` 保持可读，仅用于历史复核。

这个包只修训练压力与 artifact provenance，不改变默认 production wiring，也不改变
P4/P3 结论。下一份 routed artifact 必须重新跑 P4：门 A pass + 门 B pass 只允许说
`carrier_is_live=true`，仍不能替代 P3/P2 的行为识别与跨家族盲裁判。

#### 2026-07-28 routed 标准验证（机制通过，整体不晋升）

本次验证先确认两个工程事实：

1. 托管沙箱内 `torch.backends.mps.is_available()` 为 false；经用户启用 Metal 后，
   沙箱外 MPS 可用，因此本轮重跑 P3/P4 使用 `--device mps`；
2. `transformers` 的 fused attention 会让 `output_attentions` 为空 tuple，因此 trainer
   改为 `attn_implementation="eager"`，并对空 attentions fail loudly；否则 route
   loss 会静默退化为 0。

第一轮标准规模 routed bake 使用初始默认 `route_weight=0.35`：

```bash
python scripts/train_state_kv_prefix.py --device cpu \
  --states 16 --epochs 3 --slots 4 --rank 4 --norm-cap 0.2 \
  --output artifacts/state_kv/projectors/qwen2.5-0.5b-routed-prefix.json
```

产物 `qwen2.5-0.5b-routed-prefix.json` 的 artifact ID 为
`e0bb9cea41828ae217c5a1d984ccc45d4a4053ad59d97257740ccf24cecb0871`，
manifest 记录 `route_weight=0.35`、`route_temperature=0.18`、
`base_model_mutated=false`。训练后 wrong-user control 为 68/128 =
**0.531**，只比随机略高。

P4 标准诊断：

```bash
python scripts/run_state_kv_carrier_diagnostics.py --device cpu \
  --train-states 96 --eval-states 32 --shuffle-draws 5 \
  --prefix-kv-artifact artifacts/state_kv/projectors/qwen2.5-0.5b-routed-prefix.json \
  --output artifacts/state_kv/p4-routed/verdict_carrier_diagnostics.json
```

| 门 | 状态 | 数字 |
|---|---|---|
| A（注意力被读） | **fail** | 跨 slot 区分度 20/24 层超过随机对照（需 >12），但 state spread **0.01544** 仍低于 sentence spread **0.01706** |
| B（线性可读出） | **pass** | layer 3 held-out mean R² **0.9532**；打乱零分布上界 0.1107；无前缀对照 −0.0521 |
| overall | `carrier_is_live=false` | |

这说明 routed 目标方向正确但训练压力不足：slot 已经能区分，但注意力剖面仍被探针句
牵动得更多。随后将默认 route 权重调到 **1.0** 并重新 bake：

```bash
python scripts/train_state_kv_prefix.py --device cpu \
  --states 16 --epochs 3 --slots 4 --rank 4 --norm-cap 0.2 \
  --route-weight 1.0 \
  --output artifacts/state_kv/projectors/qwen2.5-0.5b-routed-prefix-rw1.json
```

产物 `qwen2.5-0.5b-routed-prefix-rw1.json` 的 artifact ID 为
`a8dbb43380a56b6fc6b70eae640bd56a3dce9738e97bd88b8f4e5a279c49c17b`。
manifest 记录 `training_mode=teacher-distilled-routed-prefix-v1`、
`route_weight=1.0`、`route_temperature=0.18`、`base_model_mutated=false`。
训练后 wrong-user control 为 67/128 = **0.523**，仍不构成身份选择性证据。

P4 标准诊断：

```bash
python scripts/run_state_kv_carrier_diagnostics.py --device cpu \
  --train-states 96 --eval-states 32 --shuffle-draws 5 \
  --prefix-kv-artifact artifacts/state_kv/projectors/qwen2.5-0.5b-routed-prefix-rw1.json \
  --output artifacts/state_kv/p4-routed-rw1/verdict_carrier_diagnostics.json
```

| 门 | 状态 | 数字 |
|---|---|---|
| A（注意力被读） | **pass** | 跨 slot 区分度 20/24 层超过随机对照（需 >12）；state spread **0.02066** vs sentence spread **0.01781** |
| B（线性可读出） | **pass** | layer 12 held-out mean R² **0.9617**；打乱零分布上界 0.0971；无前缀对照 −0.0521 |
| overall | `carrier_is_live=true` | |

这证明 `route_weight=1.0` 修掉了旧 artifact 的 Gate A 机制失败：State-KV carrier
现在在标准 P4 上被模型读到，且状态可以从真实 prompt token 表示中读出。

P3 标准行为识别：

```bash
python scripts/run_state_kv_identification.py \
  --lane p3 --device cpu --max-new-tokens 32 \
  --personal-conditioning-scale 0.12 \
  --prefix-kv-artifact artifacts/state_kv/projectors/qwen2.5-0.5b-routed-prefix-rw1.json \
  --output artifacts/state_kv/p3-routed-rw1/verdict_identification.json
```

| 项 | 结果 |
|---|---|
| `claim_prompt_identity` | **pass**：12 turns across 3 probes share one prompt_fp per probe with `prompt_state_sections=0` |
| `claim_output_divergence` | **pass**：G 臂每个 delivered turn 均注入，且每个 probe 的双人输出分叉 |
| C5 | **decode-matched** |
| claim 3 / 4 | `insufficient_data`：未接跨家族盲裁判 |
| overall | `insufficient_data` |

因此这条 teacher-distilled routed artifact 能升级的主张只有：**`route_weight=1.0`
的 routed State-KV carrier 已经在标准机制门上 live，并能在 P3 中产生
prompt-closed 输出分叉**。仍不能说身份识别已被证明，更不能把它描述成完整产品默认
路径已搞定；它的文本教师目标在 blind judge 上没有形成足够身份选择性。

#### 这两个门能说什么，不能说什么

**能**：这条载体是活的——状态到得了 prefill，也能从真实 token 的表示里线性读出。

**不能**：门 B 通过**不代表状态被使用**。探针测的是"在不在"，不是"起不起作用"；
而且生成器本身近似线性，高 R² 有相当一部分只是反映了这一点。行为层 P3 仍是
“是否被输出使用”的晋升门。

#### Blind judge 接线与裁判选择（2026-07-28）

`packages/vz-runtime/src/volvence_zero/state_kv_blind_judge.py` 现在提供两类本地盲裁判，
都在构造期强制跨模型家族，且 `match(*, response_text, candidate_user_ids)` 不接收臂标签、
prompt、指纹或内部状态向量：

- `LocalTransformersBlindJudge`：causal-LM letter-logprob 裁判，顺序对称化，全程 greedy。
  实测 TinyLlama-1.1B-Chat 在本任务上退化为全 tie（30/30 ties），A/B token logprob
  完全相同，因此只能作为“裁判不可用”的否证记录，不能用于通过 P3。
- `LocalEmbeddingBlindJudge`：embedding-cosine 裁判，素材与候选用户同源，使用 mean-pool
  `last_hidden_state` 后 L2 归一化，按 response 与两个候选素材的余弦相似度决策。
  当前正式 smoke P3 使用 `BAAI/bge-m3`（HF `model_type=xlm-roberta`）裁判
  Qwen substrate（`model_type=qwen2`），满足跨家族纪律。

P3 探针从 3 条扩展到 6 条，不放松 `ci_low > 0.5`，而是增加二选一 vote 数使
bootstrap CI 有能力区分 10/12 这类结果。

#### State-strategy routed Prefix-KV 标准验证（2026-07-28）

为避免 B′ 文本教师把目标限制在“复述 rendered statement”的天花板，trainer 新增
`state-strategy-routed-prefix-v1`：目标文本由 16 维状态与探针句确定性生成，直接要求
模型输出与状态相配的策略姿态，例如高 overwhelm / 低 control 时进入稳态、修复和小步
行动，高 stability / 高 trust / 高 readiness 时进入标准、下一步和验证推进。
这仍只消费 `PersonalConditioningSnapshot` 的 16 维 typed readout，不新增语义 owner，
不读原始对话，也不改基底权重。

先用 8 states / 2 epochs smoke artifact 验证方向后，本节晋升到 16 states / 3 epochs
标准 bake：

```bash
python scripts/train_state_kv_prefix.py --device mps \
  --states 16 --epochs 3 --max-new-tokens 48 \
  --route-weight 1.0 \
  --output artifacts/state_kv/projectors/qwen2.5-0.5b-state-strategy-routed-prefix.json
```

产物 artifact ID 为
`8064f8b6de8ec215807619f404c84404087109076634d1ffda53112b4684e238`，
manifest 记录 `training_mode=state-strategy-routed-prefix-v1`、
`target_source=state-strategy`、`state_count=16`、`epochs=3`、
`sample_count=128`、`route_weight=1.0`、`wrong_user_control_accuracy=0.875`、
`base_model_mutated=false`。

P3 行为识别（MPS + embedding judge）：

```bash
python scripts/run_state_kv_identification.py \
  --lane p3 --device mps --max-new-tokens 48 \
  --personal-conditioning-scale 0.12 \
  --prefix-kv-artifact artifacts/state_kv/projectors/qwen2.5-0.5b-state-strategy-routed-prefix.json \
  --judge-kind embedding --judge-model-id BAAI/bge-m3 --judge-device mps \
  --output artifacts/state_kv/p3-state-strategy-routed/verdict_identification.json
```

| 项 | 结果 |
|---|---|
| overall | **`retain-strict`** |
| `claim_prompt_identity` | **pass**：24 turns across 6 probes share one `prompt_fp` per probe，`prompt_state_sections=0` |
| `claim_output_divergence` | **pass**：G prefix arm 每个 delivered turn 均注入，且每个 probe 双人输出分叉 |
| `claim_identification_above_chance` | **pass**：12/12，accuracy **1.000**，CI **1.000..1.000**，judge `BAAI/bge-m3` |
| `claim_carrier_causality` | **pass**：A-pure 6/12，CI 0.250..0.750 覆盖 chance；G prefix 12/12，CI low 1.000 清 chance |
| C5 | **decode-matched** |

P4 机制诊断（同一 artifact）：

```bash
python scripts/run_state_kv_carrier_diagnostics.py --device mps \
  --train-states 96 --eval-states 32 --shuffle-draws 5 \
  --prefix-kv-artifact artifacts/state_kv/projectors/qwen2.5-0.5b-state-strategy-routed-prefix.json \
  --output artifacts/state_kv/p4-state-strategy-routed/verdict_carrier_diagnostics.json
```

| 门 | 状态 | 数字 |
|---|---|---|
| A（注意力被读） | **pass** | 跨 slot 区分度 20/24 层超过随机对照（需 >12）；state spread **0.02340** vs sentence spread **0.00935** |
| B（线性可读出） | **pass** | layer 23 held-out mean R² **0.9141**；打乱标签上界 0.0811；无前缀对照 −0.0521 |
| overall | `carrier_is_live=true` | |

因此当前可以严格说：**在这个标准 artifact 上，16 维 state readout 已经通过
Prefix-KV 进入冻结 Qwen 的 prefill，被 attention 读取，并在 prompt-closed、
decode-matched 的 P3 行为识别中被跨家族 embedding judge 认出。** 还不能说默认产品
路径已切到 State-KV，也不能说多 seed 复核或 rollout gate 已完成；这些仍是晋升门。

#### P2 held-out pairwise retain（2026-07-28）

P2 不复用 P3 的两个评测 persona，也不复用 P3 的 6 条 probe。runner 新增
`--lane p2` 与 `--p2-pair`：P2 只消费 held-out persona pair 与 held-out probe set，
训练脚本的 `_assert_probe_holdout()` 同时检查训练探针与 P3/P2 探针均不相交。由于
当前 `LocalEmbeddingBlindJudge` 是二选一裁判，P2 以 pairwise verdict 形式发布；
每个 pair 是一次完整识别实验，不把四个 held-out persona 混成同一多分类任务。

两个 pair 覆盖不同状态轴：

| pair | 对照语义 |
|---|---|
| `repair-vs-execute` | 高压力 / 高修复 / 高可逆性 vs 高稳定 / 高信任 / 高执行准备 |
| `boundary-vs-commit` | 高边界风险 / 高自主性风险 / 高回撤需求 vs 高承诺 / 高推进 / 低风险 |

命令：

```bash
python scripts/run_state_kv_identification.py \
  --lane p2 --p2-pair repair-vs-execute --device mps \
  --max-new-tokens 48 --personal-conditioning-scale 0.12 \
  --prefix-kv-artifact artifacts/state_kv/projectors/qwen2.5-0.5b-state-strategy-routed-prefix.json \
  --judge-kind embedding --judge-model-id BAAI/bge-m3 --judge-device mps \
  --output artifacts/state_kv/p2-state-strategy-routed-repair-vs-execute/verdict_identification.json

python scripts/run_state_kv_identification.py \
  --lane p2 --p2-pair boundary-vs-commit --device mps \
  --max-new-tokens 48 --personal-conditioning-scale 0.12 \
  --prefix-kv-artifact artifacts/state_kv/projectors/qwen2.5-0.5b-state-strategy-routed-prefix.json \
  --judge-kind embedding --judge-model-id BAAI/bge-m3 --judge-device mps \
  --output artifacts/state_kv/p2-state-strategy-routed-boundary-vs-commit/verdict_identification.json
```

结果：

| pair | overall | G-prefix matching | A-pure control | C5 |
|---|---|---|---|---|
| `repair-vs-execute` | **`retain-strict`** | 29/32，accuracy 0.906，CI **0.781..1.000** | 16/32，CI 0.312..0.688 覆盖 chance | `decode-matched` |
| `boundary-vs-commit` | **`retain-strict`** | 27/32，accuracy 0.844，CI **0.719..0.969** | 16/32，CI 0.344..0.688 覆盖 chance | `decode-matched` |

这把 P2 的 held-out 主张闭合到以下范围：同一标准 artifact、同一冻结 Qwen、同一
prompt-suppressed / decode-matched 纪律下，未参与训练的 persona 与 probe 仍能被跨家族
embedding judge 显著认回；且 A-pure 控制臂保持随机区间，排除 prompt 残留和裁判
单纯利用探针句的解释。

仍不能外推到默认产品路径或长期稳定性：P2 当前是 deterministic 单配置、pairwise
二选一，不是多 seed / 多 rollout / 多模型裁判矩阵。默认 `ACTIVE` 晋升前还需要
把 seed/rollout 稳定性、rollout gate 与回滚开关作为独立证据包关闭。

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

`prompt_state_delivery` 默认 `"text"`，为现状行为的逐字节等价路径；三个 pure 臂是显式 profile，
不进任何默认矩阵。`personal_conditioning_carrier` 默认 `"residual"`；省略
`personal_conditioning_prefix` 参数即回滚 Prefix-KV 载体，已加载但未被请求的
artifact 对默认路径完全惰性（有守门测试）。整包 `git revert` 即回滚，无状态迁移。
