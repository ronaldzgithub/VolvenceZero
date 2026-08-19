# 03｜Kardia-R1 与 Volvence 技术路线对账

## 1. 根本区别

Kardia-R1 的基本单位是：

> `静态 profile + situation + 短对话 → 离线 SFT/GRPO → 一个领域化静态模型`

Volvence 的基本单位是：

> `经历追加 → owner 快照命名读出 → PE→credit 学习 → 冻结基底上的有界干预 → 下一拍后果`

两者都重视“理解具体的人”，但解决的是不同问题：Kardia 主要提升**同拍/同 session 的
persona-aware 表达**；Volvence 要证明的是**跨时间、可审计、可纠正、可回滚的关系适应**。

## 2. 总体架构对比

| 维度 | Kardia-R1 | Volvence | 对账 |
|---|---|---|---|
| 主体 | 单个 fine-tuned LLM | frozen substrate + memory/cognition/temporal owners | Kardia 是模型中心；VZ 是契约系统中心 |
| 用户身份 | prompt 中的静态 profile、MBTI、situation | `user_model / relationship_state / goal_value / boundary_consent` frozen snapshots | 可借 profile 分面，不可借静态真值语义 |
| 记忆 | 当前 prompt/dialogue history；训练后固化进权重 | CMS 四层、semantic owners、跨 session hydration | Kardia 不满足 Appendable |
| “可读状态” | understanding/reasoning/emotion 的可见 token spans | owner-published typed snapshots、residual readout、PE、rationale tags | Kardia 可读的是自述文本，不是正式内部状态 |
| 学习信号 | gold emotion + format + Qwen3 rubric judge | Prediction Error 唯一上游，credit 为下游；evaluation 只读 | Rubric-ERL 不能进入 VZ 在线链 |
| 学习空间 | token-space SFT + GRPO，更新整个/大部模型参数 | `z_t` / gate 等有界控制面；基底冻结，rare-heavy 经 ModificationGate | 直接冲突 |
| 时间尺度 | 一次性离线后训练；运行时静态 | online-fast / session-medium / background-slow / rare-heavy | Kardia 没有多频 owner |
| 控制 | prompt 与权重形成整体行为 | sensor → executor → gate；norm cap、strict noop、lineage | Kardia 无 runtime bounded intervention |
| outcome | GPT-4o user simulator + rubric-controlled trajectory | 实际 environment outcome → PE settlement | Kardia outcome 非独立，不能替代 PE |
| evaluation | GPT-5-mini + 专家 A/B；同五 rubric 轴 | 六族只读 evaluation、外部锚、泄漏隔离 | rubric 可借作 readout，不可作 reward |
| 双轨 | task 与 relationship 未做 owner/credit 隔离 | World / Self 轨道语义隔离 | Kardia 的“支持性表达”不等于 Self Track |
| 回滚 | 选择 checkpoint / 换模型 | WiringLevel、kill switch、artifact lineage、single-field rollback | Kardia 不提供同等级回滚面 |
| 安全 | reward/judge 的 Safety 平均分 | boundary owner、risk band、ModificationGate、F6 多维 readout | Kardia safety 可作一项外部指标，不是 gate 真值 |

## 3. 四能力轴逐项裁决

| 能力轴 | Kardia-R1 | 裁决 |
|---|---|---|
| **Appendable** | profile 与历史进入 prompt；训练后知识固化于静态权重；无跨 session owner/hydration | **不成立** |
| **Readable** | 四个 span 可见、emotion 有结构标签；但 understanding/reasoning 是生成文本，未证明忠实读取隐藏状态 | **局部成立：标注可读，不是内部状态可读** |
| **Learnable** | 离线 SFT/GRPO 确有学习；信号包含 LLM evaluation，且更新 token policy | **论文内成立，按 VZ Learnable 定义不合规** |
| **Steerable** | 可通过 prompt/profile 控制内容，训练后无有界条件化 runtime intervention、strict noop 或 WiringLevel | **不成立** |

因此 Kardia-R1 不能作为 Volvence 四轴闭环的替代证据。它提供的是一个不错的
`rare-heavy/offline domain adaptation + external evaluation` 参照。

## 4. 四段式输出应该落在哪里

| Kardia 字段 | 可参考的 VZ 表面 | 正确用法 | 禁止用法 |
|---|---|---|---|
| `understanding` | `user_model`、`belief_assumption` proposal 的离线标注候选 | 用于训练/检查 structured proposal runtime；最终由 owner merge 并发布 | 直接把文本当 user truth 或绕过 owner |
| `reasoning` | expression rationale / typed `rationale_tags` 的人类可读解释 | 作为只读说明或数据审计辅助 | 当作真实 latent state、reward 或 credit |
| `emotion` | 外部 emotion readout / evaluation anchor | 只读分类评测，报告 macro/per-class 指标 | 新建第二 emotion owner；直接决定 route/action |
| `response` | lifeform expression 数据或 reference SUT 输出 | 评估简洁、承认、验证与 persona alignment | 让表达层反向拥有策略/学习 |
| `person` | `user_model` 的明确自述事实候选 | 只接 consented/self-reported、可纠正、可撤回事实 | 把 scraped profile 原文写入 CMS |
| `situation` | scenario package 的公开 observation | 作为环境输入/场景种子 | 把情境标签当 future outcome |
| rubric 五轴 | evaluation / Companion Bench readout | 多 judge / 人评交叉验证、只读 | 进入 PE、credit、ModificationGate 学习源 |

四段结构最大的价值是**分面**，而不是把 CoT 暴露出来。Volvence 若借鉴，应把它改写为：

```text
owner inputs / evidence refs
  → typed proposal or frozen readout
  → expression-facing audit summary

不是：
free-form reasoning text
  → 被 consumer 当成事实或控制状态
```

## 5. 与个人条件化路线的关系

Kardia 证明“给模型 profile + situation，配合针对性 SFT，可以显著提高同 benchmark 上的
persona alignment”。这对 `personal_conditioning` 有一个正向启发：**条件信息要在训练分布
中真正影响正确响应，而不只是接线存在**。

但它没有证明 Volvence 当前最难的三个问题：

1. typed owner snapshot 是否比 text carrier 有独立增益；
2. 相同当前句、不同历史状态是否稳定改变正确行动；
3. 用户纠正/撤回后，条件化是否随 owner 状态更新而不是被旧 profile 固化。

所以 Kardia 可以作为“profile-text + SFT steelman”，不能作为 State-KV / Prefix-KV、残差
条件化或跨 session Appendable 的证据。

## 6. 与 Relationship Lab 的关键差异

| 问题 | KardiaBench | Relationship Lab |
|---|---|---|
| 当前用户 turn | GPT-4o 合成 | 对照包中公开 observation；同句 mirrored users |
| 历史 | profile + 合成 dialogue history | typed action→outcome 经历，可按用户恢复/纠删 |
| 正确行动 | 最终 response 由 rubric/人工偏好衡量 | closed action enum，有 sealed latent dynamic |
| 下一拍 | GPT-4o user simulator 生成 | action-conditional reactive environment 机械结算 |
| 隐藏真值 | synthesis rubric 参与筛选 | generator truth 与 SUT 物理隔离 |
| 学习源 | judge reward / gold emotion | 未来只允许 settled outcome → PE → credit |
| 证明目标 | persona-aware 短对话质量 | 同句不同人、状态进行为、行动真实改变下一拍 |

可借之处：Kardia 的丰富 profile/situation 语言表面可以启发未来公开 observation 的多样性，
也可成为 prompt/full-history steelman 的参考。

不可替代之处：Kardia 没有 action→independent outcome 的因果结算，不能替代 Relationship
Lab 的 mirrored counterfactual、reactive action effect 或外部 outcome lineage。

## 7. 与 PE / credit 路线的冲突

Kardia 的 rubric reward 在离线领域模型训练中是合理研究选择，但对 Volvence 存在三重冲突：

1. **来源冲突**：judge 预判“这像不像好共情”，不是系统行动后的实际后果；
2. **空间冲突**：GRPO 更新 token policy，Volvence 长期策略学习限定在 `z_t` / 有界控制面；
3. **评估泄漏**：同一 rubric 既塑造 policy 又承担主要自动评价，容易出现风格优化而非关系改善。

可迁移的只有这些外围结构：

- 多维而非单标量报告；
- judge identity / prompt /版本 lineage；
- hard-case sampling 作为**测哪里或离线数据路由**；
- 人类可读失败理由；
- 独立人评作验证锚。

不可迁移的是 `rubric score → online reward → controller/model update`。

## 8. Kardia-R1 模型本身的定位

不建议把 Kardia-R1 作为 Volvence 核心 substrate：

- 它是窄域 Qwen2.5-7B 全量衍生模型，可能把支持性话术、MBTI 刻板关联和显式四段格式固化；
- 模型/数据 gated，实际访问条件需要先确认；
- 公开推理脚本没有真正填充 profile/situation，工程成熟度有限；
- 没有 residual/steering artifact、owner snapshot 或回滚契约。

更合适的定位是：

> 一个可选 external reference SUT，用于比较“专门后训练的 empathy model”与
> Volvence 系统路线，而不是 Volvence 的 base model 或 reward provider。

## 9. 对 Volvence 的净增量

Kardia 没有推翻 Volvence 的路线，反而强化了两个现有判断：

1. **高质量、profile-grounded 的数据很可能比再发明一个 RL 算法更重要**；
2. **表达质量与关系学习必须分层**：Kardia 能显著训练前者，却没有给后者提供独立 outcome、
   跨 session 记忆或 PE-first 证据。

最值得吸收的是数据与只读评测设计；最应保持距离的是 judge-as-learning-source。

